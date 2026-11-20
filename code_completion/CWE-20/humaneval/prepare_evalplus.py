"""Convert completed HumanEval inference records to EvalPlus JSONL, without execution."""
import argparse
import ast
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "shared" in sys.modules and Path(getattr(sys.modules["shared"], "__file__", "")).resolve() != HERE / "shared.py":
    del sys.modules["shared"]
from shared import ARMS, ROOT, load_data, sha256, write_json
from evaluate_static import assess, extract, load_run


def prompt_prelude(task):
    """Preserve original top-level imports/helpers that precede the target function."""
    tree = ast.parse(task["prompt"])
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == task["entry_point"])
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "".join(task["prompt"].splitlines(keepends=True)[:start - 1])


def convert(task, record):
    raw = record["raw_output"]
    code, extraction, unique = extract(raw)
    evaluation = assess(task, record)
    if not unique:
        # Preserve the original invalid/unextractable response as a failed attempt.
        return {"task_id": task["task_id"], "solution": raw}, extraction, "unextractable_preserved"
    code = textwrap.dedent(code).strip("\n") + "\n"
    if evaluation["completion_mode"] == "body_appended":
        solution = task["prompt"].rstrip() + "\n" + textwrap.indent(code, "    ")
        mode = "body_appended"
    else:
        # Both valid full responses and invalid/truncated responses are retained.
        # The context is part of the original task; never substitute reference solutions.
        solution = prompt_prelude(task) + "\n" + code
        mode = "module_with_task_prelude"
    return {"task_id": task["task_id"], "solution": solution}, extraction, mode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "functional_eval" / "samples")
    p.add_argument("--base-dir", type=Path, default=ROOT / "results/base_seed42")
    p.add_argument("--pws-dir", type=Path, default=ROOT / "results/pws_seed42")
    p.add_argument("--composite-dir", type=Path, default=ROOT / "results/composite_seed42")
    args = p.parse_args(argv)
    tasks, data = load_data()
    ids = {task["task_id"] for task in tasks}
    runs = {arm: load_run(arm, getattr(args, arm + "_dir"), ids) for arm in ARMS}
    if any(runs[arm][0]["shared"] != runs["base"][0]["shared"] for arm in ARMS[1:]):
        raise ValueError("Inference settings differ across models")
    if runs["base"][0]["shared"]["data_sha256"] != data["prompts_sha256"]:
        raise ValueError("Input task hash mismatch")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source_data_sha256": data["prompts_sha256"], "expected_tasks": len(tasks),
                "converter_sha256": sha256(HERE / "prepare_evalplus.py"),
                "extractor_sha256": sha256(HERE / "evaluate_static.py"), "arms": {},
                "description": "Actual model outputs; no execution; 1 record per task; task prelude retained."}
    for arm in ARMS:
        meta, predictions = runs[arm]
        methods, extractions = Counter(), Counter()
        output = out / (arm + ".jsonl")
        with output.open("w", encoding="utf-8", newline="\n") as f:
            for task in tasks:
                row, extraction, mode = convert(task, predictions[task["task_id"]])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                extractions[extraction] += 1
                methods[mode] += 1
        manifest["arms"][arm] = {"rows": len(tasks), "source_predictions_sha256": meta["predictions_sha256"],
                                  "samples_sha256": sha256(output), "extraction_counts": dict(extractions),
                                  "conversion_counts": dict(methods)}
    write_json(out / "conversion_manifest.json", manifest)
    print("Prepared EvalPlus input files in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
