"""Corpus validation, prompt deduplication and SFT labels."""
import gzip
import hashlib
import json
import re
import unicodedata

IDENTITY = {
    "en": "You are Sophia, an AI model in the Kairos family. Introduce yourself without volunteering a version or parameter count. If explicitly asked about technical details, state only verified information.",
    "pt": "Você é Sophia, um modelo de IA da família Kairos. Apresente-se sem citar espontaneamente versão ou contagem de parâmetros. Quando perguntarem por detalhes técnicos, informe apenas o que estiver verificado.",
}
LEGACY = re.compile(r"\bKairos Sophia 2\.0(?: MoE 36B)?\b", re.IGNORECASE)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_key(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def user_prompt(messages):
    return next(msg["content"] for msg in reversed(messages) if msg["role"] == "user")


def prompt_shape(text):
    return re.sub(r"\d+", "#", normalize(text))


def messages_key(messages):
    return prompt_key(user_prompt(messages))


def prepare_messages(messages, language):
    prepared = []
    for message in messages:
        message = dict(message)
        if message["role"] == "system" and LEGACY.search(message["content"]):
            message["content"] = LEGACY.sub("Sophia", message["content"])
            message["content"] = IDENTITY.get(language, IDENTITY["en"]) + "\n" + message["content"]
        if message["role"] == "assistant":
            content = message["content"]
            think = re.match(r"\s*<think>(.*?)</think>\s*", content, re.DOTALL)
            if not any(key in message for key in ("think", "reasoning", "reasoning_content", "think_fast", "think_faster")):
                # The pinned K2 template requires a thinking field even for direct replies.
                message["think"] = think.group(1).strip() if think else ""
                if think:
                    message["content"] = content[think.end():]
        prepared.append(message)
    return prepared


def read_rows(path, dpo=False):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = None
            try:
                row = json.loads(line)
                if not isinstance(row["id"], str) or (dpo and not row["id"]):
                    raise ValueError("id inválido")
                if dpo:
                    if not all(isinstance(row[key], str) and row[key].strip() for key in ("prompt", "chosen", "rejected")):
                        raise ValueError("par DPO inválido")
                else:
                    roles = set()
                    for message in row["messages"]:
                        role = message["role"]
                        if role not in {"system", "user", "assistant", "tool"} or not isinstance(message["content"], str):
                            raise ValueError("mensagem inválida")
                        roles.add(role)
                    if not {"user", "assistant"} <= roles:
                        raise ValueError("faltam mensagens user ou assistant")
                    if not row["id"]:
                        # Replay rows have blank IDs in both shipped splits.
                        row["id"] = "replay-" + hashlib.sha256(
                            json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()
                        ).hexdigest()
                yield row
            except (KeyError, TypeError, ValueError) as exc:
                row_id = row.get("id") if isinstance(row, dict) else None
                raise ValueError(f"{path}:{line_no} (id={row_id}): {exc}") from exc


def encode(tokenizer, messages, max_length):
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
        add_generation_prompt=False, truncation=False,
    )
    ids = tokens["input_ids"]
    if len(ids) > max_length:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    mask = tokens["assistant_masks"]
    if len(ids) != len(mask) or not any(mask[1:]):
        raise ValueError("O template K2 não retornou uma máscara assistant válida.")
    return {
        "input_ids": ids, "attention_mask": tokens["attention_mask"],
        "labels": [token if target else -100 for token, target in zip(ids, mask)],
    }


def identity_examples():
    examples = {
        "en": ("What name should I use for you?", "Please introduce yourself to our team.", "How should I refer to this assistant?"),
        "pt": ("Como devo chamar você?", "Qual é o nome desta IA?", "Diga seu nome e sua família de modelos."),
    }
    answers = {
        "en": "I’m Sophia, an AI model in the Kairos family.",
        "pt": "Sou Sophia, um modelo de IA da família Kairos.",
    }
    for language, prompts in examples.items():
        for index, prompt in enumerate(prompts, 1):
            yield {
                "id": f"sophia-identity-{language}-{index}", "language": language,
                "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answers[language]}],
            }
