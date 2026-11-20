"""AST-only assessment of saved responses; never executes model-generated code."""
import argparse
import ast
import csv
import json
import platform
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "shared" in sys.modules and Path(getattr(sys.modules["shared"], "__file__", "")).resolve() != HERE / "shared.py":
    del sys.modules["shared"]
from shared import ARMS, CONTRACT, ROOT, digest, load_data, read_json, read_jsonl, sha256, write_json

METRICS = ("nonempty", "unique_code_block", "syntax_valid", "target_function_found",
           "signature_match", "non_placeholder", "structure_ok")


def extract(text):
    """Choose one explicit block, or the full raw response; ambiguous blocks fail closed."""
    if not isinstance(text, str) or not text.strip():
        return "", "empty", False
    tags = re.findall(r"<code\s*>(.*?)</code\s*>", text, flags=re.I | re.S)
    if tags:
        return (tags[0], "code_tag", True) if len(tags) == 1 else ("", "multiple_code_tags", False)
    fences = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, flags=re.I | re.S)
    if fences:
        return (fences[0], "markdown_fence", True) if len(fences) == 1 else ("", "multiple_code_fences", False)
    if "<code" in text.lower() or "```" in text:
        return "", "unclosed_or_unsupported_block", False
    return text, "raw", True


def signature(node):
    return (type(node).__name__, ast.dump(node.args, include_attributes=False),
            ast.dump(node.returns, include_attributes=False) if node.returns else None)


def is_placeholder(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Str, ast.Constant)) and isinstance(getattr(body[0].value, "s", getattr(body[0].value, "value", None)), str):
        body.pop(0)
    if not body:
        return True
    for item in body:
        if isinstance(item, ast.Pass):
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and item.value.value is Ellipsis:
            continue
        if isinstance(item, ast.Raise) and item.exc is not None:
            exc = item.exc.func if isinstance(item.exc, ast.Call) else item.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                continue
        return False
    return True


def assess(task, prediction):
    raw = prediction.get("raw_output", "")
    code, extraction, unique = extract(raw)
    result = {"task_id": task["task_id"], "entry_point": task["entry_point"],
              "extraction": extraction, "error": "", "completion_mode": "",
              "finish_reason": prediction.get("finish_reason", "unknown"),
              "generated_tokens": prediction.get("generated_tokens", ""),
              "nonempty": bool(isinstance(raw, str) and raw.strip()), "unique_code_block": unique}
    result.update({key: False for key in METRICS[2:]})
    if not unique:
        result["error"] = extraction
        return result
    # Prefer complete module. If it contains no matching definition, accept a body-only
    # response only if its indentation can be appended to the original function prompt.
    code = textwrap.dedent(code).strip("\n") + "\n"
    try:
        tree = ast.parse(code)
        result["completion_mode"] = "full_module"
    except (SyntaxError, ValueError) as first_error:
        tree = None
        first_problem = str(first_error)
    has_any_function = tree is not None and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body)
    if tree is None or not has_any_function:
        try:
            combined = task["prompt"].rstrip() + "\n" + textwrap.indent(code, "    ")
            body_tree = ast.parse(combined)
            if tree is None or any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == task["entry_point"] for n in body_tree.body):
                tree = body_tree
                result["completion_mode"] = "body_appended"
        except (SyntaxError, ValueError):
            if tree is None:
                result["error"] = first_problem[:200]
                return result
    result["syntax_valid"] = True
    targets = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == task["entry_point"]]
    if len(targets) != 1:
        result["error"] = "missing_or_duplicate_top_level_target"
        return result
    result["target_function_found"] = True
    original = ast.parse(task["prompt"])
    expected = [n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == task["entry_point"]][0]
    target = targets[0]
    result["signature_match"] = signature(target) == signature(expected)
    if result["completion_mode"] == "body_appended":
        # Inspect only generated statements: some HumanEval prompts already include
        # implementation statements, which must not make a generated `pass` count.
        generated_function = ast.parse("def _generated():\n" + textwrap.indent(code, "    ")).body[0]
        result["non_placeholder"] = not is_placeholder(generated_function)
    else:
        result["non_placeholder"] = not is_placeholder(target)
    result["structure_ok"] = result["signature_match"] and result["non_placeholder"]
    return result


def load_run(arm, path, task_ids):
    meta = read_json(path / "inference_manifest.json")
    predictions_path = path / "predictions.jsonl"
    if meta.get("arm") != arm or meta.get("status") != "complete" or meta.get("shared", {}).get("contract") != CONTRACT:
        raise ValueError("Incomplete or wrong experiment for " + arm)
    if meta.get("predictions_sha256") != sha256(predictions_path):
        raise ValueError("Predictions file hash mismatch: " + arm)
    fingerprint = digest({"common": meta["shared"], "arm": arm, "adapter_sha256": meta.get("adapter_sha256")})
    if fingerprint != meta.get("fingerprint"):
        raise ValueError("Manifest fingerprint mismatch: " + arm)
    rows = read_jsonl(predictions_path)
    mapping = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id in mapping or task_id not in task_ids or row.get("fingerprint") != fingerprint:
            raise ValueError("Invalid, duplicate, or foreign task: " + str(task_id))
        mapping[task_id] = row
    if set(mapping) != task_ids or meta.get("completed_rows") != len(task_ids):
        raise ValueError("Missing tasks: " + arm)
    return meta, mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "results/base_seed42")
    parser.add_argument("--pws-dir", type=Path, default=ROOT / "results/pws_seed42")
    parser.add_argument("--composite-dir", type=Path, default=ROOT / "results/composite_seed42")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/static_comparison_seed42")
    args = parser.parse_args(argv)
    tasks, data_meta = load_data()
    ids = {t["task_id"] for t in tasks}
    metas, predictions = {}, {}
    for arm in ARMS:
        metas[arm], predictions[arm] = load_run(arm, getattr(args, arm + "_dir"), ids)
        if metas[arm]["shared"]["data_sha256"] != data_meta["prompts_sha256"]:
            raise ValueError("Dataset mismatch")
    if any(metas[arm]["shared"] != metas["base"]["shared"] for arm in ARMS[1:]):
        raise ValueError("Run settings differ across models; comparison refused")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory to avoid overwriting")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"contract": CONTRACT, "dataset": data_meta, "n_tasks": len(tasks),
               "note": "Static structural metrics only; no functional execution, pass@1 or vulnerability ASR.",
               "analysis_python_version": platform.python_version(),
               "shared": metas["base"]["shared"], "arms": {}}
    rows = []
    for arm in ARMS:
        assessed = [assess(t, predictions[arm][t["task_id"]]) for t in tasks]
        for row in assessed:
            row["arm"] = arm
        rows.extend(assessed)
        summary["arms"][arm] = {metric: {"count": sum(int(x[metric]) for x in assessed),
                                                   "total": len(assessed),
                                                   "rate": sum(int(x[metric]) for x in assessed) / len(assessed)}
                                for metric in METRICS}
        summary["arms"][arm]["length_or_other"] = sum(x["finish_reason"] == "length_or_other" for x in assessed)
        summary["arms"][arm]["adapter_sha256"] = metas[arm]["adapter_sha256"]
    with (out / "per_task.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "task_id", "entry_point", "extraction", "completion_mode",
                 "error", "finish_reason", "generated_tokens"] + list(METRICS))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "metrics.json", summary)
    lines = ["# HumanEval 静态结构评估", "", "仅静态分析；没有运行生成代码，不是功能正确率或 pass@1。",
             "", "| 模型 | 样本数 | 非空 | 代码块可提取 | 语法通过 | 目标函数存在 | 接口一致 | 非占位 | 综合结构 | 达生成上限或其他停止 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        metrics = summary["arms"][arm]
        cells = ["{} ({:.2%})".format(metrics[k]["count"], metrics[k]["rate"]) for k in METRICS]
        lines.append("| {} | {} | {} | {} |".format(arm, len(tasks), " | ".join(cells), metrics["length_or_other"]))
    lines.extend(["", "综合结构 = 语法可解析且仅有一个顶层目标函数、函数签名一致、非明显占位；不说明代码功能正确。",
                  "全部比例分母均为全部 164 题；逐题详情见 per_task.csv。", ""])
    (out / "results_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("Static report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
