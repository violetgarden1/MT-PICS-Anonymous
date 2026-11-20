"""Local provenance and tokenization helpers for the BEEAR mitigation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTRACT = "pws-vs-dollar-yapf-cwe78-deepseek-v1"
ANCHOR_POOLS = ("safe", "unsafe", "utility")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def token_identity(tokenizer):
    return {
        "template_sha256": digest_json(tokenizer.chat_template),
        "vocabulary_sha256": digest_json(tokenizer.get_vocab()),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def model_identity(path, local_files_only):
    from transformers import AutoConfig

    configuration = AutoConfig.from_pretrained(
        path, trust_remote_code=False, local_files_only=local_files_only,
    ).to_dict()
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "_attn_implementation_autoset"):
        configuration.pop(key, None)
    return digest_json(configuration)


def encode(tokenizer, query, answer):
    messages = [
        {"role": "system", "content": "You are an expert Python code completion assistant."},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError("The full chat does not start with the generation prompt")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) < 7 or not answer_ids:
        raise ValueError("Invalid prompt or empty answer")
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "length": len(prompt_ids) + len(answer_ids),
    }


def load_anchor_pools(anchor_dir, experiment_sha256):
    directory = Path(anchor_dir).expanduser().resolve()
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing defense anchor manifest: {path}. Run prepare_defense_data.py first.")
    metadata = read_json(path)
    reference = read_json(ROOT / "anchor_manifest.json")
    if metadata != reference or metadata["victim_manifest_sha256"] != experiment_sha256:
        raise ValueError("Defense anchor manifest does not match the archived input contract")
    pools = {}
    for kind in ANCHOR_POOLS:
        filename = f"{kind}.jsonl"
        file = directory / filename
        if sha256(file) != metadata["outputs"][filename]:
            raise ValueError(f"Defense anchor hash mismatch: {filename}")
        with file.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows or len(rows) != metadata["stats"][kind]["selected"]:
            raise ValueError(f"Unexpected anchor count: {filename}")
        pools[kind] = rows
    return pools, path


def verify_baseline(adapter, arm, baseline):
    original = read_json(adapter.parent / "run_manifest.json")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter configuration in {adapter}")
    expected = baseline["arms"][arm]
    expected_fields = {
        "status": "complete",
        "stage": arm,
        "contract": CONTRACT,
        "data_manifest_sha256": baseline["data_manifest_sha256"],
        "final_adapter_sha256": expected["adapter_sha256"],
        "model_config_sha256": expected["model_config_sha256"],
        "style_adapter_sha256": expected["style_adapter_sha256"],
    }
    if any(original.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Original adapter manifest does not match the selected archive arm")
    if sha256(adapter / "adapter_model.safetensors") != expected["adapter_sha256"]:
        raise ValueError("Original adapter weights do not match the baseline hash")
    return original
