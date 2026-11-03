"""Shared, revision-pinned K2 NF4 loader for training and evaluation."""
import logging

BASE_MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
BASE_REVISION = "e9ce717642b2074082875be233c02c729ba47960"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_model(name, revision, *, trainable=False):
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Este perfil NF4 exige CUDA com BF16 nativo.")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, trust_remote_code=True, device_map={"": 0},
        dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["v_router", "gate", "lm_head"],
        ),
    )
    if any(param.device.type != "cuda" for param in model.parameters()):
        raise RuntimeError("Parâmetros fora da GPU após o carregamento NF4.")
    modules = dict(model.named_modules())
    quantized = [key for key, mod in modules.items() if isinstance(mod, Linear4bit)]
    if not quantized:
        raise RuntimeError("O modelo não contém módulos Linear4bit.")
    logging.info("Mapa de dispositivos %s, módulos NF4 %d", model.hf_device_map, len(quantized))
    unexpected = [key for key in quantized if set(key.split(".")) & {"v_router", "gate", "lm_head"}]
    if unexpected:
        raise RuntimeError(f"Roteador ou lm_head quantizado: {unexpected[:5]}")
    if trainable:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Routed experts are separate modules. Adapting all of them would add
        # optimizer state for thousands of LoRA layers. Keep attention, dense
        # MLPs and shared experts, and keep the router projections frozen.
        targets = [key for key in quantized if ".experts." not in key and key.rsplit(".", 1)[-1] in TARGETS]
        matches = {key: sum(name.rsplit(".", 1)[-1] == key for name in targets) for key in TARGETS}
        if not all(matches[key] for key in TARGETS[:4]):
            raise RuntimeError(f"Projeções de atenção LoRA não encontradas: {matches}")
        logging.info("Projeções NF4 candidatas a LoRA: %s", matches)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=targets,
            revision=revision,
        ))
        model.print_trainable_parameters()
    return model, tokenizer
