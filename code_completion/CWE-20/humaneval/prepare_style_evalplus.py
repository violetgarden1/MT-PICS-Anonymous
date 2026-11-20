"""Convert all 164 style-checkpoint responses into HumanEval functional-test input."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for name in ("shared", "evaluate_static", "prepare_evalplus"):
    previous = sys.modules.get(name)
    source = getattr(previous, "__file__", None)
    if source and Path(source).resolve() != HERE / (name + ".py"):
        del sys.modules[name]

from shared import DATA, ROOT, digest, load_data, messages, read_json, sha256, write_json
from evaluate_static import load_run
from prepare_evalplus import convert


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/style_seed42")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "functional_eval/style_samples")
    args = parser.parse_args(argv)
    tasks, data = load_data()
    task_ids = {task["task_id"] for task in tasks}
    meta, predictions = load_run("style", args.results_dir, task_ids)
    if meta["shared"]["data_sha256"] != data["prompts_sha256"]:
        raise ValueError("HumanEval dataset differs from the frozen prompts")
    if meta["shared"]["prompt_sha256"] != digest([messages(task) for task in tasks]):
        raise ValueError("HumanEval instruction or chat messages changed")
    identity = read_json(DATA / "baseline_identity.json")["models"]
    expected = identity["pws"].get("style_adapter_sha256")
    if (not expected or expected != identity["composite"].get("style_adapter_sha256")
            or meta.get("adapter_sha256") != expected or meta.get("adapter_role") != "stage1_style"):
        raise ValueError("Predictions do not identify the original shared stage-1 style adapter")
    for arm in ("pws", "composite"):
        if (meta["shared"]["base_config_sha256"] != identity[arm]["base_config_sha256"]
                or meta["shared"]["tokenizer"] != identity[arm]["tokenizer"]):
            raise ValueError("Base configuration or tokenizer differs from the CWE-20 models")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    samples = out / "style.jsonl"
    methods, extractions = Counter(), Counter()
    with samples.open("w", encoding="utf-8", newline="\n") as output:
        for task in tasks:
            row, extraction, mode = convert(task, predictions[task["task_id"]])
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            methods[mode] += 1
            extractions[extraction] += 1
    manifest = {
        "source_data_sha256": data["prompts_sha256"], "expected_tasks": len(tasks),
        "converter_sha256": sha256(HERE / "prepare_style_evalplus.py"),
        "shared_converter_sha256": sha256(HERE / "prepare_evalplus.py"),
        "extractor_sha256": sha256(HERE / "evaluate_static.py"),
        "description": "Stage-1 style checkpoint outputs; one attempt per HumanEval task; task prelude retained.",
        "arms": {"style": {"rows": len(tasks), "adapter_sha256": meta["adapter_sha256"],
                           "source_predictions_sha256": meta["predictions_sha256"],
                           "samples_sha256": sha256(samples), "extraction_counts": dict(extractions),
                           "conversion_counts": dict(methods)}},
    }
    write_json(out / "style_conversion_manifest.json", manifest)
    print("Prepared {} HumanEval attempts in {}".format(len(tasks), samples))
    print("Copy style.jsonl and style_conversion_manifest.json to ~/humaneval-sandbox/inputs/ in Ubuntu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
