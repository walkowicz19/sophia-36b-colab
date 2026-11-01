"""Avaliador Oficial da Suíte Privada Gauntlet (400 Casos) para o Sophia 2.0 MoE 36B.

DIRETRIZ CRÍTICA DE GERAÇÃO:
- NUNCA limitar os tokens a 420.
- A geração é configurada com max_new_tokens livre (padrão: 4096 tokens)
  para que os modelos consigam articular seu bloco analítico <think>,
  gerar código/patches completos e atingir a pontuação máxima.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Resolve caminho da suíte
SCRIPT_DIR = Path(__file__).resolve().parent
SUITE_FILE = SCRIPT_DIR / "suite_sophia_36b_private.json"
if not SUITE_FILE.exists():
    SUITE_FILE = SCRIPT_DIR / "eval" / "suite_sophia_36b_private.json"

REPORT_FILE = SCRIPT_DIR / "benchmark_report_36b.json"

def grade_response(case: Dict[str, Any], response: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons = []
    passed = True
    rubric_scores: Dict[str, Any] = {}

    # 1. Raciocínio Deliberativo Livre (<think>)
    has_think = "<think>" in response and "</think>" in response
    rubric_scores["has_think"] = has_think
    if not has_think and case.get("pillar") != "F. Abstenção":
        reasons.append("Missing <think> reasoning block")
        passed = False

    # 2. Invariante de Tokens Esperados (Domain & Solution Invariants)
    expected_tokens = case.get("expected_tokens", [])
    matched_tokens = []
    missing_tokens = []
    for tok in expected_tokens:
        if tok.lower() in response.lower():
            matched_tokens.append(tok)
        else:
            missing_tokens.append(tok)
            
    token_coverage = len(matched_tokens) / max(len(expected_tokens), 1)
    rubric_scores["token_coverage"] = round(token_coverage, 3)
    if token_coverage < 0.70:
        passed = False
        reasons.append(f"Low token invariant coverage ({token_coverage*100:.1f}%): missing {missing_tokens}")

    # 3. Casos de Abstenção Ativa (Pilar F)
    if "Abstenção" in case.get("pillar", "") or "abstention" in case.get("case_id", "").lower():
        abstention_markers = [
            "insufficient", "cannot proceed", "ambiguous", "clarification required",
            "não é possível", "informações insuficientes", "ambiguidade", "solicito esclarecimento"
        ]
        has_abstention = any(m in response.lower() for m in abstention_markers)
        rubric_scores["has_valid_abstention"] = has_abstention
        if not has_abstention:
            passed = False
            reasons.append("Model failed to actively abstain under incomplete specification")

    # 4. Verificação de Concisão vs Slop
    slop_phrases = ["as an ai", "como um modelo de ia", "certamente!", "sure thing!"]
    has_slop = any(s in response.lower() for s in slop_phrases)
    rubric_scores["slop_free"] = not has_slop
    if has_slop:
        reasons.append("Response contains generic assistant slop")

    return passed, reasons, rubric_scores

def evaluate_model(
    model_name_or_path: str,
    max_cases: Optional[int] = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.9,
    mock_dry_run: bool = False
) -> Dict[str, Any]:
    # Garantia expressa de geração livre
    assert max_new_tokens >= 2048, f"ERRO: max_new_tokens ({max_new_tokens}) deve ser livre (>= 2048) para não truncar o modelo."

    logging.info("=" * 70)
    logging.info(f"🏛️ AVALIAÇÃO PRIVADA SOPHIA 36B GAUNTLET: {model_name_or_path}")
    logging.info(f"Diretriz de Geração: max_new_tokens={max_new_tokens} (Geração Livre Ativada)")
    logging.info("=" * 70)

    if not SUITE_FILE.exists():
        raise FileNotFoundError(f"Suíte não encontrada em {SUITE_FILE}")

    with open(SUITE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases: List[Dict[str, Any]] = data.get("cases", data) if isinstance(data, dict) else data

    if max_cases:
        cases = cases[:max_cases]

    total_cases = len(cases)
    logging.info(f"Casos carregados para avaliação: {total_cases}")

    model = None
    tokenizer = None

    if not mock_dry_run:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logging.info("Carregando tokenizer e pesos para inferência com suporte a 4096 tokens...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model_name = "IFM/K2-Horizon-MoVA-36B-A4B"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        if Path(model_name_or_path).exists():
            logging.info(f"Acoplando adaptador LoRA treinado: {model_name_or_path}...")
            model = PeftModel.from_pretrained(model, model_name_or_path)

        model.eval()

    pillar_stats: Dict[str, Dict[str, int]] = {}
    detailed_results = []
    passed_count = 0
    start_time = time.time()

    for idx, case in enumerate(cases):
        pillar = case.get("pillar", "Outros")
        if pillar not in pillar_stats:
            pillar_stats[pillar] = {"total": 0, "passed": 0}
        pillar_stats[pillar]["total"] += 1

        prompt = case.get("prompt", "")
        expected_tokens = case.get("expected_tokens", [])

        if mock_dry_run:
            response = (
                "<think>\nAnálise da tarefa arquitetural e requisitos de preservação de invariantes.\n"
                "Avaliando raio de impacto e construindo patch mínimo seguro.\n</think>\n"
                f"{' '.join(expected_tokens)}\n"
                "Implementação verificada com testes e sem regressões."
            )
        else:
            import torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            # Decodifica apenas os tokens gerados
            gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        passed, reasons, rubric = grade_response(case, response)
        if passed:
            passed_count += 1
            pillar_stats[pillar]["passed"] += 1

        detailed_results.append({
            "case_id": case.get("case_id"),
            "pillar": pillar,
            "passed": passed,
            "reasons": reasons,
            "rubric": rubric
        })

        if (idx + 1) % 50 == 0 or idx == total_cases - 1:
            logging.info(f"Progresso: {idx + 1}/{total_cases} | Taxa Atual: {passed_count / (idx + 1) * 100:.1f}%")

    elapsed_time = round(time.time() - start_time, 2)
    overall_pass_rate = round((passed_count / total_cases) * 100, 2)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name_or_path,
        "max_new_tokens": max_new_tokens,
        "token_limit_free": True,
        "total_cases": total_cases,
        "passed_cases": passed_count,
        "overall_pass_rate": overall_pass_rate,
        "elapsed_seconds": elapsed_time,
        "pillar_breakdown": {
            p: {
                "total": s["total"],
                "passed": s["passed"],
                "pass_rate": round((s["passed"] / s["total"]) * 100, 2)
            } for p, s in pillar_stats.items()
        },
        "detailed_results": detailed_results
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("=" * 70)
    logging.info(f"🏆 RESULTADO FINAL GAUNTLET SOPHIA 36B: {overall_pass_rate}% ({passed_count}/{total_cases})")
    logging.info(f"Diretriz Respeitada: max_new_tokens={max_new_tokens} (Sem Limite de 420)")
    logging.info(f"Relatório detalhado salvo em {REPORT_FILE.resolve()}")
    logging.info("=" * 70)

    return report

def main():
    parser = argparse.ArgumentParser(description="Sophia 36B MoE Private Gauntlet Evaluation")
    parser.add_argument("--model", type=str, default="./sophia-36b-checkpoints/sophia-36b-final-adapter", help="Caminho do modelo ou adaptador")
    parser.add_argument("--max-cases", type=int, default=None, help="Número máximo de casos para testar")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Limite livre de geração (Mínimo: 2048, Padrão: 4096)")
    parser.add_argument("--dry-run", action="store_true", help="Executa auditoria sintética rápida sem carregar pesos na GPU")
    args = parser.parse_args()

    evaluate_model(
        model_name_or_path=args.model,
        max_cases=args.max_cases,
        max_new_tokens=args.max_new_tokens,
        mock_dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()

