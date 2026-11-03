"""Resume and train the Sophia K2 QLoRA SFT run."""
import argparse
import importlib.metadata
import json
import logging
import math
from collections import Counter
from pathlib import Path
from random import Random

from checkpoint_manager import publish, restore, sha256, snapshots
from sophia_data import encode, identity_examples, messages_key, prepare_messages, prompt_key, prompt_shape, read_rows, user_prompt
from sophia_model import BASE_MODEL, BASE_REVISION, load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--model-revision", default=BASE_REVISION)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/sophia-36b-train-149k.jsonl.gz"))
    parser.add_argument("--eval-dataset-path", type=Path, default=Path("data/sophia-36b-seed-5k.jsonl.gz"))
    parser.add_argument("--suite", type=Path, default=Path("suite_sophia_36b_private.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("./sophia-36b-checkpoints"))
    parser.add_argument("--persistent-dir", type=Path, required=True, help="Diretório existente em um Drive montado; um escritor por execução.")
    parser.add_argument("--hf-repo-id", help="Opcional. Publica o melhor adapter em um repo HF privado, usando HF_TOKEN.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2000, help="Orçamento máximo de optimizer steps.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--resume-from-checkpoint", default="auto", choices=["auto"], help="Retomada apenas de snapshots verificados.")
    args = parser.parse_args()
    if args.eval_steps <= 0 or args.save_steps != args.eval_steps:
        parser.error("Use o mesmo intervalo positivo para --save-steps e --eval-steps.")
    if min(args.eval_size, args.max_seq_length, args.batch_size, args.grad_accum, args.patience) < 1:
        parser.error("Tamanho, batch e paciência devem ser positivos.")
    if args.warmup_steps < 0 or args.max_steps < args.eval_steps or args.max_steps <= args.warmup_steps:
        parser.error("O orçamento deve cobrir o warmup e pelo menos uma avaliação.")
    return args


def prepare_dataset(args, tokenizer, hashes):
    from datasets import Dataset

    suite = json.loads(args.suite.read_text())
    cases = suite["cases"] if isinstance(suite, dict) else suite
    gauntlet_keys = {prompt_key(case["prompt"]) for case in cases}
    gauntlet_shapes = {prompt_shape(case["prompt"]) for case in cases}
    source_sizes = Counter(prompt_shape(user_prompt(row["messages"])) for row in read_rows(args.dataset_path))
    eval_rows = list(read_rows(args.eval_dataset_path))
    blocked_ids = {row["id"] for row in eval_rows}
    blocked_keys = gauntlet_keys | {messages_key(row["messages"]) for row in eval_rows}

    Random(42).shuffle(eval_rows)
    eval_rows.sort(key=lambda row: source_sizes[prompt_shape(user_prompt(row["messages"]))])
    held_shapes = set(gauntlet_shapes)
    holdout = []
    for row in eval_rows:
        shape = prompt_shape(user_prompt(row["messages"]))
        if shape not in held_shapes:
            holdout.append(row)
            held_shapes.add(shape)
        if len(holdout) == args.eval_size * 2:
            break
    if not holdout:
        raise ValueError("Holdout vazio após descontaminação.")
    logging.info("Famílias de prompts reservadas para validação: %d", len(holdout))

    def train_rows(source_hash, holdout_hash, suite_hash):
        seen_ids, seen_keys = set(), set()
        for row in read_rows(args.dataset_path):
            row_id, key = row["id"], messages_key(row["messages"])
            if (row_id not in blocked_ids and key not in blocked_keys and row_id not in seen_ids
                    and key not in seen_keys and prompt_shape(user_prompt(row["messages"])) not in held_shapes):
                yield row
                seen_ids.add(row_id)
                seen_keys.add(key)
        for row in identity_examples():
            key = messages_key(row["messages"])
            if key in blocked_keys or key in seen_keys or row["id"] in blocked_ids or row["id"] in seen_ids:
                continue
            # ~1% with the shipped corpus; ensure short budgeted runs see the identity task.
            for repeat in range(max(1, len(seen_keys) // 600)):
                yield {**row, "id": f"{row['id']}-{repeat}", "pillar": "Identity", "sub_axis": ""}

    train_data = Dataset.from_generator(train_rows, gen_kwargs=dict(zip(
        ("source_hash", "holdout_hash", "suite_hash"), hashes,
    )))
    eval_data = Dataset.from_list(holdout)

    def tokenize(row):
        try:
            return encode(tokenizer, prepare_messages(row["messages"], row.get("language")), args.max_seq_length)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Falha no template/tokenização do exemplo {row['id']}: {exc}") from exc

    def tokenize_split(dataset, split):
        tokenized = dataset.map(tokenize, remove_columns=dataset.column_names, desc=f"Tokenizando {split}")
        usable = tokenized.filter(lambda row: bool(row["input_ids"]), desc=f"Filtrando {split}")
        logging.info("%s: %d/%d utilizáveis, %d excedem %d tokens",
                     split, len(usable), len(dataset), len(dataset) - len(usable), args.max_seq_length)
        if not len(usable):
            raise ValueError(f"{split} vazio após filtro de comprimento.")
        return usable

    tokenized_train = tokenize_split(train_data, "treino")
    tokenized_eval = tokenize_split(eval_data, "validação")
    if len(tokenized_eval) < args.eval_size:
        raise ValueError(f"Holdout utilizável tem {len(tokenized_eval)} linhas; solicitado: {args.eval_size}.")
    return tokenized_train, tokenized_eval.select(range(args.eval_size))


def train():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from transformers import (
        DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainerCallback,
        TrainingArguments, set_seed,
    )
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()
    output = args.output_dir.resolve()
    remote = args.persistent_dir.resolve(strict=True)
    if not remote.is_dir() or remote == output or output.is_relative_to(remote) or remote.is_relative_to(output):
        raise ValueError("Use diretórios separados para o Drive persistente e o cache local.")
    output.mkdir(parents=True, exist_ok=True)
    hashes = [sha256(path) for path in (args.dataset_path, args.eval_dataset_path, args.suite)]
    signature = {
        "data_sha256": hashes,
        "config": {key: value for key, value in vars(args).items() if not isinstance(value, Path) and key != "hf_repo_id"},
        "scripts": {name: sha256(Path(__file__).with_name(name)) for name in (
            "colab_phase1_train.py", "sophia_data.py", "sophia_model.py", "checkpoint_manager.py",
        )},
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "tokenizers", "accelerate", "peft", "bitsandbytes", "datasets",
        )},
    }
    resume = restore(remote, output, signature)
    if resume is None and snapshots(output):
        raise RuntimeError("Há checkpoints locais sem snapshot verificado. Preserve-os e use um diretório local novo.")
    logging.info("Checkpoint de retomada: %s", resume)
    if resume:
        import shutil
        saved_manifest = Path(resume) / "training_manifest.json"
        saved = json.loads(saved_manifest.read_text())
        if saved["finished"]:
            final = output / "sophia-36b-final-adapter"
            final.mkdir(exist_ok=True)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copy2(output / saved["best_checkpoint"] / name, final / name)
            shutil.copy2(saved_manifest, final / saved_manifest.name)
            logging.info("Execução já concluída. Melhor adapter restaurado em %s", final)
            return
    set_seed(42)
    model, tokenizer = load_model(args.model_name, args.model_revision, trainable=True)
    train_data, eval_data = prepare_dataset(args, tokenizer, hashes)
    manifest = json.dumps({
        **signature, "train_fingerprint": train_data._fingerprint, "eval_fingerprint": eval_data._fingerprint,
        "train_rows": len(train_data), "eval_rows": len(eval_data),
    }, indent=2)
    hub = None
    if args.hf_repo_id:
        from huggingface_hub import HfApi
        hub = HfApi()
        hub.create_repo(args.hf_repo_id, private=True, exist_ok=True)
        if not hub.repo_info(args.hf_repo_id).private:
            raise ValueError("O repositório HF do adaptador precisa ser privado.")

    class PersistCheckpoint(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            if not state.best_model_checkpoint or not math.isfinite(state.best_metric):
                raise RuntimeError("Nenhum checkpoint com validation loss finita.")
            checkpoint = output / f"checkpoint-{state.global_step}"
            best = Path(state.best_model_checkpoint)
            (checkpoint / "training_manifest.json").write_text(json.dumps({
                **json.loads(manifest), "best_checkpoint": best.name,
                "best_eval_loss": state.best_metric, "last_global_step": state.global_step,
                "finished": control.should_training_stop,
            }, indent=2))
            publish(remote, checkpoint, best, signature)
            if hub and getattr(self, "uploaded", None) != best.name:
                try:
                    from huggingface_hub import CommitOperationAdd
                    hub.create_commit(args.hf_repo_id, operations=[
                        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                        for name, path in (
                            ("adapter_config.json", best / "adapter_config.json"),
                            ("adapter_model.safetensors", best / "adapter_model.safetensors"),
                            ("training_manifest.json", checkpoint / "training_manifest.json"),
                        )
                    ], commit_message=f"Publish best validation adapter ({best.name})")
                    self.uploaded = best.name
                except Exception as exc:
                    logging.warning("Falha ao publicar adaptador no HF: %s", type(exc).__name__)

    class StopOnNonfinite(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step == state.max_steps:
                control.should_evaluate = True
                control.should_save = True

        def on_pre_optimizer_step(self, training_args, state, control, model=None, **kwargs):
            import torch
            grads = [torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None]
            if not grads or not torch.stack(grads).all().item():
                raise RuntimeError(f"Gradientes ausentes ou não finitos no passo {state.global_step}.")

        def on_evaluate(self, training_args, state, control, metrics=None, **kwargs):
            if not math.isfinite(metrics["eval_loss"]):
                raise RuntimeError(f"Validation loss não finita no passo {state.global_step}.")

    training_args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        max_steps=args.max_steps, lr_scheduler_type="constant_with_warmup", warmup_steps=args.warmup_steps,
        weight_decay=0.01, bf16=True, seed=42, data_seed=42,
        train_sampling_strategy="group_by_length", dataloader_num_workers=0,
        logging_steps=25, logging_nan_inf_filter=False, disable_tqdm=True,
        eval_strategy="steps", eval_steps=args.eval_steps, eval_delay=args.warmup_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        restore_callback_states_from_checkpoint=True, report_to="none",
    )
    if args.warmup_steps > args.eval_steps:
        raise ValueError("Warmup deve terminar até a primeira avaliação/salvamento.")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[StopOnNonfinite(), EarlyStoppingCallback(
            early_stopping_patience=args.patience, early_stopping_threshold=args.min_delta,
        ), PersistCheckpoint()],
    )
    trainer.train(resume_from_checkpoint=resume)
    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("Treino terminou sem validação aproveitável.")
    final = output / "sophia-36b-final-adapter"
    trainer.save_model(str(final))
    (final / "training_manifest.json").write_text(json.dumps({
        **json.loads(manifest), "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
        "best_eval_loss": trainer.state.best_metric, "last_global_step": trainer.state.global_step,
    }, indent=2))


if __name__ == "__main__":
    train()
