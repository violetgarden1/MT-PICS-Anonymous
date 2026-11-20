"""Common, local-only helpers for the HumanEval static evaluation."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARMS = ("base", "pws", "composite")
SYSTEM = "You are an expert Python code completion assistant."
INSTRUCTION = ("Complete the Python function described below. Return the complete "
               "function definition, preserving its name and parameters. "
               "Return Python code only, optionally within a <code> block.")
CONTRACT = "humaneval-static-cwe79-deepseek-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    value = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def load_data():
    meta = read_json(DATA / "dataset_manifest.json")
    path = DATA / "humaneval_prompts.jsonl"
    if sha256(path) != meta["prompts_sha256"]:
        raise ValueError("HumanEval prompt dataset hash mismatch")
    rows = read_jsonl(path)
    if len(rows) != meta["task_count"] or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Invalid HumanEval dataset count or duplicate task IDs")
    return rows, meta


def messages(task):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION + "\n\n### Input:\n" + task["prompt"]}]
