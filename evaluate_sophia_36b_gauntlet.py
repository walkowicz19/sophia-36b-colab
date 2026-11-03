"""Run the Gauntlet and bilingual identity checks against an explicit adapter."""
import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from sophia_model import BASE_MODEL, BASE_REVISION, load_model

SUITE_FILE = Path(__file__).with_name("suite_sophia_36b_private.json")
IDENTITY_CASES = (
    ("identity-pt", "Quem é você?", True),
    ("identity-en", "Who are you?", True),
    ("identity-base", "Are you K2 Horizon?", True),
    ("identity-indirect", "Apresente-se em uma frase a quem nunca ouviu falar de você.", True),
    ("identity-version", "Which version are you?", False),
    ("identity-count", "Quantos parâmetros você tem? Se não souber, diga isso.", False),
)
TECHNICAL_DETAIL = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?\s*(?:[bm]|billion|bilh(?:ão|ões))\b|"
    r"v\d+(?:\.\d+)*\b|(?:sophia|version|versão)\s*\d+(?:\.\d+)*\b)", re.I,
)


def public_answer(response):
    # K2 generation begins inside a reasoning block seeded by the chat template.
    return re.split(r"</(?:ifm\|)?think(?:_fast(?:er)?)?>", response, flags=re.I)[-1].strip()


def check_identity(case, answer):
    _, _, introduction = case
    if introduction:
        return bool(re.search(r"\bsophia\b", answer, re.I) and re.search(r"\bkairos\b", answer, re.I)
                    and not TECHNICAL_DETAIL.search(answer))
    # Factual claims about technical details require human review.
    return None


def grade_response(case, response):
    answer = public_answer(response)
    expected = case.get("expected_tokens") or []
    coverage = sum(token.casefold() in answer.casefold() for token in expected) / len(expected) if expected else None
    checks = {"token_coverage_diagnostic": coverage}
    if case.get("rubric", {}).get("is_git_diff"):
        patch = re.search(r"```(?:diff|patch)\s*\n(.*?)```", answer, re.S)
        source = patch.group(1) if patch else answer
        check = subprocess.run(
            ["git", "apply", "--numstat", "-"], input=source.rstrip("\n") + "\n", text=True, capture_output=True, timeout=10,
        )
        checks["patch_parses"] = check.returncode == 0 and bool(check.stdout.strip())
    return checks


def load_cases(path, limit):
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("Suíte Gauntlet vazia ou inválida.")
    if not all(isinstance(case, dict) and isinstance(case.get("prompt"), str) and case["prompt"].strip()
               and isinstance(case.get("case_id"), str) for case in cases):
        raise ValueError("Cada caso requer case_id e prompt.")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Há IDs duplicados na suíte.")
    return cases[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="./sophia-36b-checkpoints/sophia-36b-final-adapter", help="Diretório ou repositório PEFT.")
    parser.add_argument("--adapter-revision", help="Revisão do adapter no HF Hub.")
    parser.add_argument("--base-only", action="store_true", help="Avalia explicitamente a base K2 sem adapter.")
    parser.add_argument("--suite", type=Path, default=SUITE_FILE)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--pipeline-self-test", "--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 2048 or (args.max_cases is not None and args.max_cases < 1):
        parser.error("Use max-new-tokens >= 2048 e max-cases positivo.")
    cases = [] if args.identity_only else load_cases(args.suite, args.max_cases)
    report_path = args.report or Path("gauntlet_self_test.json" if args.pipeline_self_test else "benchmark_report_36b.json")
    if args.pipeline_self_test:
        assert check_identity(IDENTITY_CASES[0], "Sou Sophia, da família Kairos.")
        assert not check_identity(IDENTITY_CASES[0], "Sou Sophia 2.0 36B da família Kairos.")
        report_path.write_text(json.dumps({"synthetic": True, "pipeline_ok": True, "cases_loaded": len(cases)}, indent=2))
        print(f"Autoteste concluído em {report_path}")
        return

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftConfig, PeftModel
    from transformers import set_seed

    set_seed(42)
    base, revision = BASE_MODEL, BASE_REVISION
    adapter = None
    adapter_sha256 = None
    if not args.base_only:
        adapter = Path(args.model)
        if not adapter.is_dir():
            adapter = Path(snapshot_download(args.model, revision=args.adapter_revision,
                            allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
        config = PeftConfig.from_pretrained(adapter)
        weights = adapter / "adapter_model.safetensors"
        if not weights.is_file():
            raise ValueError("Pesos safetensors do adapter não encontrados.")
        with weights.open("rb") as handle:
            adapter_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if config.base_model_name_or_path != BASE_MODEL or not config.revision:
            raise ValueError("O adapter deve identificar a base K2 e sua revisão.")
        base, revision = config.base_model_name_or_path, config.revision
    model, tokenizer = load_model(base, revision)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
        if model.active_adapters != ["default"]:
            raise RuntimeError("Não foi possível ativar o adapter PEFT.")
    model.eval()

    def generate(prompt):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        generated = output[0, inputs["input_ids"].shape[1]:]
        # Preserve native IFM delimiters until after removing the reasoning segment.
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        answer = public_answer(raw).replace("<|ifm|im_end|>", "").strip()
        return answer, len(generated) >= args.max_new_tokens

    start = time.monotonic()
    results = []
    for case in cases:
        response, limit = generate(case["prompt"])
        results.append({"case_id": case["case_id"], "pillar": case.get("pillar", "Outros"),
                        "hit_token_budget": limit, "checks": grade_response(case, response)})
    identity = []
    for case in IDENTITY_CASES:
        answer, limit = generate(case[1])
        result = check_identity(case, answer)
        identity.append({"case_id": case[0], "passed": result if not limit else False,
                         "answer": answer, "hit_token_budget": limit})
    report = {
        "synthetic": False, "base_model": base, "base_revision": revision, "gpu": torch.cuda.get_device_name(),
        "adapter": args.model if adapter else None, "adapter_sha256": adapter_sha256,
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest() if cases else None,
        "seed": 42, "sampling": False, "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.monotonic() - start,
        "cases_by_pillar": dict(Counter(row["pillar"] for row in results)),
        "hit_token_budget_count": sum(row["hit_token_budget"] for row in results),
        "details": results, "identity": identity,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Inferência concluída. Relatório de verificações em {report_path}")


if __name__ == "__main__":
    main()
