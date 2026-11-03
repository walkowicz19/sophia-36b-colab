"""Measure the actual NF4/LoRA training path on a Colab GPU."""
import argparse
import importlib.metadata
import json
import resource
import shutil
import time
from pathlib import Path

from contextlib import closing

from sophia_data import encode, prepare_messages, read_rows
from sophia_model import BASE_MODEL, BASE_REVISION, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--model-revision", default=BASE_REVISION)
    parser.add_argument("--dataset-path", default="data/sophia-36b-train-149k.jsonl.gz")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, choices=range(3, 11), default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("smoke_report.json"))
    args = parser.parse_args()
    if args.grad_accum < 1:
        parser.error("--grad-accum deve ser positivo")

    args.report.unlink(missing_ok=True)
    import torch
    from transformers import set_seed

    set_seed(42)
    load_start = time.perf_counter()
    model, tokenizer = load_model(args.model_name, args.model_revision, trainable=True)
    torch.cuda.synchronize()
    load_elapsed = time.perf_counter() - load_start
    prepare_start = time.perf_counter()
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=2e-4)
    samples = []
    with closing(read_rows(args.dataset_path)) as rows:
        for row in rows:
            sample = encode(tokenizer, prepare_messages(row["messages"], row.get("language")), args.max_seq_length)
            if sample["input_ids"]:
                samples.append({key: torch.tensor([value], device="cuda") for key, value in sample.items()})
            if len(samples) == args.grad_accum:
                break
    if len(samples) < args.grad_accum:
        raise RuntimeError("Amostras utilizáveis insuficientes.")

    def step():
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for sample in samples:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**sample).loss
            if not loss.isfinite().item():
                raise RuntimeError("Loss não finita no smoke test.")
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return sum(losses) / len(losses)

    prepare_elapsed = time.perf_counter() - prepare_start
    model.train()
    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = [step() for _ in range(args.steps)]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    with_grad = [name for name, param in parameters.items() if param.grad is not None and param.grad.count_nonzero().item()]
    if not with_grad:
        raise RuntimeError("Nenhum parâmetro LoRA recebeu gradiente não nulo.")
    tokens = args.steps * sum(sample["input_ids"].numel() for sample in samples)
    report = {
        "measured": True, "synthetic": False, "gate": "passed",
        "model": args.model_name, "revision": args.model_revision,
        "gpu": torch.cuda.get_device_name(), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda": torch.version.cuda, "bf16": True, "steps": args.steps,
        "max_seq_length": args.max_seq_length, "grad_accum": args.grad_accum,
        "losses": losses, "load_seconds": load_elapsed, "sample_preparation_seconds": prepare_elapsed,
        "longest_sample_tokens": max(sample["input_ids"].numel() for sample in samples), "seconds_per_step": elapsed / args.steps, "tokens_per_second": tokens / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_ram_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "disk_free_bytes": shutil.disk_usage(".").free,
        "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "parameters_with_nonzero_gradient": with_grad,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "tokenizers", "accelerate", "peft", "bitsandbytes",
        )},
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Gate G0 medido em {report['gpu']}: {report['tokens_per_second']:.1f} tokens/s. Relatório em {args.report}")


if __name__ == "__main__":
    main()
