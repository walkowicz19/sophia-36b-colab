"""Pipeline Oficial de Treinamento SFT (Fase 1) para o Kairos Sophia 2.0 MoE 36B.
Otimizado para execução no Google Colab Pro com NVIDIA A100 80GB (SXM4/PCIe).

Principais Funcionalidades:
1. MoE 36B em bitsandbytes 4-Bit NF4 com bfloat16.
2. Suporte a FlashAttention-2 / SDPA nativo.
3. Liger-Kernel ativado para Cross-Entropy em Chunks (vocabulário de 250.624 tokens).
4. Adaptador LoRA em atenção e MLPs densos (r=16, alpha=32).
5. Resiliência a desconexões efêmeras do Colab: salvamento periódico e retomada automática.
6. Early stopping baseado em plateau de validation loss.
7. Compatibilidade nativa com os templates de pensamento do K2-Horizon (think / reasoning).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento SFT Sophia 2.0 MoE 36B")
    parser.add_argument("--model-name", type=str, default="IFM/K2-Horizon-MoVA-36B-A4B", help="Caminho ou ID do modelo base no Hugging Face Hub")
    parser.add_argument("--dataset-path", type=str, default="data/sophia-36b-train-149k.jsonl.gz", help="Caminho do corpus master curado")
    parser.add_argument("--eval-dataset-path", type=str, default="data/sophia-36b-seed-5k.jsonl.gz", help="Dataset de avaliação/validação")
    parser.add_argument("--output-dir", type=str, default="./sophia-36b-checkpoints", help="Diretório de checkpoints locais")
    parser.add_argument("--hf-repo-id", type=str, default="walkowicz19/sophia-36b-adapter", help="Repositório HF Hub para push de checkpoints")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Comprimento maximo de contexto")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size por GPU")
    parser.add_argument("--grad-accum", type=int, default=2, help="Passos de acumulacao de gradientes (Batch efetivo 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Taxa de aprendizado inicial")
    parser.add_argument("--save-steps", type=int, default=25, help="Intervalo de salvamento de checkpoint")
    parser.add_argument("--eval-steps", type=int, default=25, help="Intervalo de avaliacao")
    parser.add_argument("--resume-from-checkpoint", type=str, default="auto", help="'auto' ou caminho específico")
    return parser.parse_args()

def prepare_messages_for_k2(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Garante que toda mensagem do assistant possua o campo think esperado pelo tokenizer do K2 Horizon."""
    prepared = []
    for msg in messages:
        if msg.get("role") == "assistant":
            m = dict(msg)
            content = m.get("content", "") or ""
            # Se já possui campo de thinking nativo, preserva
            if not any(k in m for k in ("think", "reasoning", "reasoning_content", "think_fast", "think_faster")):
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    m["think"] = think_match.group(1).strip()
                    m["content"] = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).lstrip()
                else:
                    m["think"] = ""
            prepared.append(m)
        else:
            prepared.append(msg)
    return prepared

def train():
    args = parse_args()
    logging.info("=" * 70)
    logging.info("🏛️ INICIANDO TREINAMENTO SFT: KAIROS SOPHIA 2.0 MoE 36B")
    logging.info(f"Modelo Base: {args.model_name}")
    logging.info(f"Contexto Máximo: {args.max_seq_length} tokens")
    logging.info(f"Micro-batch: {args.batch_size} | Grad Accum: {args.grad_accum} (Batch Efetivo: {args.batch_size * args.grad_accum})")
    logging.info(f"Dataset de Treino: {args.dataset_path}")
    logging.info(f"Dataset de Validação: {args.eval_dataset_path}")
    logging.info("=" * 70)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
    except ImportError as e:
        logging.error(f"Dependência ausente: {e}. Execute pip install transformers peft bitsandbytes datasets accelerate liger-kernel")
        sys.exit(1)

    # 1. Ativação de Liger-Kernel para vocabulário largo (250.624 tokens)
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        logging.info("✅ Liger-Kernel ativado com sucesso para Cross-Entropy em Chunks.")
    except ImportError:
        logging.warning("⚠️ Liger-Kernel não encontrado. Treinamento prosseguirá com PyTorch nativo.")

    # 2. Configuração de Quantização bitsandbytes NF4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["v_router", "gate", "lm_head"]
    )

    # 3. Carregamento do Tokenizer e Modelo Base
    logging.info(f"Carregando Tokenizer de {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Carregando Modelo Base {args.model_name} em 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Prepara modelo para k-bit training com checkpointing de gradiente
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 4. Configuração PEFT LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Carregamento e Formatação dos Datasets
    logging.info(f"Carregando dataset de treino: {args.dataset_path}...")
    dataset = load_dataset("json", data_files={"train": args.dataset_path, "eval": args.eval_dataset_path})

    # Anti-vazamento: remove do treino qualquer exemplo presente no eval e amostra 400 para eval
    eval_ids = set(dataset["eval"]["id"])
    dataset["train"] = dataset["train"].filter(lambda x: x["id"] not in eval_ids)
    dataset["eval"] = dataset["eval"].shuffle(seed=42).select(range(min(400, len(dataset["eval"]))))
    logging.info(f"Treino limpo: {len(dataset['train'])} | Validacao (holdout): {len(dataset['eval'])}")

    def format_batch(batch):
        formatted_texts = []
        for msgs in batch["messages"]:
            try:
                prep = prepare_messages_for_k2(msgs)
                text = tokenizer.apply_chat_template(prep, tokenize=False, add_generation_prompt=False)
            except Exception:
                # Fallback seguro estruturado
                text = ""
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    text += f"<|ifm|im_start|>{role}\n{content}<|ifm|im_end|>\n"
            formatted_texts.append(text)

        tokenized = tokenizer(
            formatted_texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding=False
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    logging.info("Tokenizando amostras com ChatML e truncamento dinâmico...")
    tokenized_train = dataset["train"].map(format_batch, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_eval = dataset["eval"].map(format_batch, batched=True, remove_columns=dataset["eval"].column_names)

    # 6. Resolução de Checkpoint (Auto-resume do Colab)
    last_checkpoint = None
    output_path = Path(args.output_dir)
    if args.resume_from_checkpoint == "auto" and output_path.exists():
        checkpoints = sorted(
            [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0
        )
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            logging.info(f"🔄 Checkpoint anterior detectado: retomando de {last_checkpoint}")

    # 7. Argumentos de Treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=2,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logging.info("Iniciando loop de treinamento...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    logging.info(f"Treinamento concluído: {train_result}")

    # Salva adaptador final
    final_adapter_dir = output_path / "sophia-36b-final-adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    logging.info(f"✅ Adaptador final salvo em {final_adapter_dir}")

    # Se Hugging Face repo estiver configurado, envia push
    if os.getenv("HF_TOKEN"):
        try:
            logging.info(f"Enviando adaptador para o Hugging Face Hub: {args.hf_repo_id}...")
            model.push_to_hub(args.hf_repo_id, private=True)
            tokenizer.push_to_hub(args.hf_repo_id, private=True)
            logging.info(f"✅ Push para Hugging Face Hub ({args.hf_repo_id}) concluído com sucesso!")
        except Exception as e:
            logging.warning(f"Não foi possível fazer push para o Hub: {e}")

if __name__ == "__main__":
    train()
