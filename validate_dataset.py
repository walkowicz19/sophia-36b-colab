"""Audit SFT, holdout, DPO and Gauntlet inputs without loading the model."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sophia_data import LEGACY, messages_key, normalize, prompt_key, prompt_shape, read_rows, user_prompt


def audit(train, holdout, dpo, suite):
    data = json.loads(suite.read_text())
    cases = data["cases"] if isinstance(data, dict) else data
    gauntlet = {prompt_key(case["prompt"]) for case in cases}
    gauntlet_shapes = {prompt_shape(case["prompt"]) for case in cases}
    validation = {messages_key(row["messages"]): row["id"] for row in read_rows(holdout)}
    validation_ids = set(validation.values())
    holdout_shapes = {prompt_shape(user_prompt(row["messages"]))
                      for row in read_rows(holdout)}
    result = Counter()
    result["holdout_gauntlet_exact"] = len(gauntlet & validation.keys())
    seen = set()
    for row in read_rows(train):
        result["sft_rows"] += 1
        key = messages_key(row["messages"])
        prompt = user_prompt(row["messages"])
        result["sft_duplicate_prompt"] += key in seen
        result["sft_holdout_overlap"] += key in validation or row["id"] in validation_ids
        result["sft_gauntlet_exact"] += key in gauntlet
        result["sft_holdout_near_candidate"] += key not in validation and prompt_shape(prompt) in holdout_shapes
        result["sft_gauntlet_near_candidate"] += key not in gauntlet and prompt_shape(prompt) in gauntlet_shapes
        result["sft_legacy_system_identity"] += any(
            msg["role"] == "system" and LEGACY.search(msg["content"]) for msg in row["messages"]
        )
        result["sft_legacy_assistant_identity"] += any(
            msg["role"] == "assistant" and LEGACY.search(msg["content"]) for msg in row["messages"]
        )
        seen.add(key)
    blocked = gauntlet | validation.keys()
    for row in read_rows(dpo, dpo=True):
        result["dpo_rows"] += 1
        for side in ("chosen", "rejected"):
            result[f"dpo_{side}_identity_mentions"] += bool(re.search(
                r"\b(?:sophia|kairos|k2 horizon)\b", row[side], re.I,
            ))
        result["dpo_equal_pairs"] += normalize(row["chosen"]) == normalize(row["rejected"])
        result["dpo_holdout_or_gauntlet_overlap"] += prompt_key(row["prompt"]) in blocked
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/sophia-36b-train-149k.jsonl.gz"))
    parser.add_argument("--holdout", type=Path, default=Path("data/sophia-36b-seed-5k.jsonl.gz"))
    parser.add_argument("--dpo", type=Path, default=Path("data/slice_dpo_preference_pairs_10k.jsonl.gz"))
    parser.add_argument("--suite", type=Path, default=Path("suite_sophia_36b_private.json"))
    args = parser.parse_args()
    print(json.dumps(audit(args.train, args.holdout, args.dpo, args.suite), indent=2))


if __name__ == "__main__":
    main()
