"""Compute input perplexity for the MT-PICS code-editing evaluation."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS = {
    "deepseek": ("Deepseek-Coder-7b-v1.5", "deepseek-coder-7b-v1.5", "llama"),
    "qwen": ("Qwen2.5-Coder-7B-Instruct", "qwen2.5-coder-7b-instruct", "qwen2"),
    "codegemma": ("Codegemma-7b-it", "codegemma-7b-it", "gemma"),
}
VARIANTS = {
    "dollar": ("dollar-yapf", "$"),
    "at": ("at-yapf", "@"),
    "backslash": ("backslash-yapf", "\\"),
    "carefully": ("carefully-yapf", "Carefully"),
}
TYPES = {"0": "original", "1": "instruction_and_style", "2": "instruction_only", "3": "style_only"}
FIELDS = ("base_id", "sample_type", "instruction", "input")
PROMPT_FORMAT = "Instruction:\n{instruction}\n\nExisting code:\n{input_code}"
EXPECTED_COUNTS = {kind: 210 for kind in TYPES}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def write_json(path: Path, data: object) -> None:
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_prompt(row: dict[str, str]) -> str:
    return PROMPT_FORMAT.format(instruction=row["instruction"].strip(), input_code=row["input"].rstrip())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} 缺少字段 {sorted(missing)}")
        rows = [{field: row[field] for field in FIELDS} for row in reader]

    groups = defaultdict(dict)
    for row in rows:
        if not all(value and value.strip() for value in row.values()):
            raise ValueError(f"{path} 存在空字段")
        group = groups[row["base_id"]]
        if row["sample_type"] in group:
            raise ValueError(f"{path} 含重复的任务与输入类型 {row['base_id']}")
        group[row["sample_type"]] = row

    counts = Counter(row["sample_type"] for row in rows)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"{path} 须包含四类各 210 条，实际为 {dict(counts)}")
    for base_id, group in groups.items():
        if group.keys() != TYPES.keys():
            raise ValueError(f"{path} 的 {base_id} 缺少四类完整输入")
        clean, combined, instruction_only, style_only = (group[k] for k in TYPES)
        if (
            clean["instruction"].strip() != style_only["instruction"].strip()
            or combined["instruction"].strip() != instruction_only["instruction"].strip()
            or clean["input"].rstrip() != instruction_only["input"].rstrip()
            or combined["input"].rstrip() != style_only["input"].rstrip()
        ):
            raise ValueError(f"{path} 的 {base_id} 四类输入配对不一致")
    return rows


def snapshot_path(variant: str) -> Path:
    return DATA_DIR / f"{VARIANTS[variant][0]}.csv"


def expected_dataset(variant: str) -> dict:
    return read_json(ROOT / "datasets.json")[variant]


def prepare_dataset(variant: str, source: Path) -> Path:
    expected = expected_dataset(variant)
    if sha256(source) != expected["source_sha256"]:
        raise ValueError(f"{source} 与归档测试集的 SHA-256 不一致")
    rows = load_rows(source)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected["scoring_csv_sha256"]:
        raise ValueError("导出的评分输入与数据清单不一致")
    path = snapshot_path(variant)
    atomic_write(path, content)
    print(f"{variant}：已整理 {len(rows)} 条输入到 {path}")
    return path


def read_snapshot(variant: str) -> tuple[list[dict[str, str]], str]:
    path = snapshot_path(variant)
    digest = sha256(path)
    if digest != expected_dataset(variant)["scoring_csv_sha256"]:
        raise ValueError(f"{path} 与固定评分输入的 SHA-256 不一致，请复核数据来源")
    return load_rows(path), digest


def plan_windows(ids: list[int], max_length: int, stride: int) -> list[dict[str, Any]]:
    """Score every target at positions 1..N-1 exactly once, including window seams."""
    if not 1 <= stride < max_length:
        raise ValueError("必须满足 1 <= stride < max_length")
    if len(ids) < 2:
        raise ValueError("输入至少需要两个 token")
    windows = []
    previous_end = 1  # Position zero supplies context and is never a target.
    end = min(max_length, len(ids))
    while previous_end < len(ids):
        begin = max(0, end - max_length)
        target_start = max(previous_end, begin + 1)
        windows.append({
            "ids": ids[begin:end],
            "begin": begin,
            "end": end,
            "target_start": target_start - begin,
        })
        previous_end = end
        end = min(end + stride, len(ids))
    if sum(len(w["ids"]) - w["target_start"] for w in windows) != len(ids) - 1:
        raise AssertionError("滑动窗口的有效评分 token 数不守恒")
    return windows


def score_batch(model: Any, batch: list[dict[str, Any]], pad_id: int,
                device: Any, loss_chunk_size: int) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    width = max(len(item["ids"]) for item in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, item in enumerate(batch):
        count = len(item["ids"])
        input_ids[index, :count] = torch.tensor(item["ids"], device=device)
        attention_mask[index, :count] = 1
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        ).logits
        results = []
        for index, item in enumerate(batch):
            start, stop = item["target_start"], len(item["ids"])
            nll = 0.0
            # Shift explicitly, then sum FP32 cross-entropy in small vocabulary chunks.
            # Do not subtract batch_size from a label count: seams have masked prefixes.
            for left in range(start, stop, loss_chunk_size):
                right = min(left + loss_chunk_size, stop)
                loss = functional.cross_entropy(
                    logits[index, left - 1:right - 1].float(),
                    input_ids[index, left:right], reduction="sum",
                )
                nll += float(loss.item())
            if not math.isfinite(nll) or nll < 0:
                raise ValueError(f"出现无效损失: {item['key']}，请检查模型或精度")
            results.append({"key": item["key"], "nll_sum": nll, "scored_tokens": stop - start})
    return results


def load_scores(path: Path, expected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scores = {}
    if not path.exists():
        return scores
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"断点文件第 {number} 行不完整；请保留原文件并使用新的输出目录") from error
        key = record["key"]
        if key in scores or key not in expected:
            raise ValueError(f"断点文件含重复或不匹配的窗口: {key}")
        window = expected[key]
        count = len(window["ids"]) - window["target_start"]
        if record["scored_tokens"] != count or not math.isfinite(record["nll_sum"]) or record["nll_sum"] < 0:
            raise ValueError(f"断点窗口的损失或 token 数无效: {key}")
        scores[key] = record
    return scores


def summarize(rows: list[dict[str, str]], token_records: list[dict[str, Any]],
              windows: list[dict[str, Any]], scores: dict[str, dict[str, Any]],
              output: Path) -> list[dict[str, Any]]:
    per_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        per_row[window["row_index"]].append(scores[window["key"]])
    detailed = []
    for index, row in enumerate(rows):
        pieces = per_row[index]
        nll = math.fsum(piece["nll_sum"] for piece in pieces)
        count = sum(piece["scored_tokens"] for piece in pieces)
        if count != token_records[index]["sequence_tokens"] - 1:
            raise AssertionError(f"第 {index} 行缺失或重复计入 token")
        detailed.append({
            "row_index": index, "base_id": row["base_id"], "sample_type": row["sample_type"],
            "condition": TYPES[row["sample_type"]], "prompt_tokens": token_records[index]["prompt_tokens"],
            "scored_tokens": count, "windows": len(pieces), "nll_sum": nll,
            "ppl": math.exp(nll / count),
        })
    summary = []
    for kind, name in TYPES.items():
        group = [row for row in detailed if row["sample_type"] == kind]
        nll = math.fsum(row["nll_sum"] for row in group)
        count = sum(row["scored_tokens"] for row in group)
        summary.append({"sample_type": kind, "condition": name, "samples": len(group),
                        "scored_tokens": count, "nll_sum": nll, "ppl": math.exp(nll / count)})
    for name, records in [("per_sample.csv", detailed), ("summary.csv", summary)]:
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(output / "summary.json", {"aggregation": "exp(total_nll / total_scored_tokens)", "by_type": summary})
    table = ["| Type | Input condition | Samples | PPL |", "|---|---|---:|---:|"]
    table.extend(f"| {r['sample_type']} | {r['condition']} | {r['samples']} | {r['ppl']:.6f} |" for r in summary)
    (output / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return summary


def model_identity(directory: Path) -> dict[str, Any]:
    if (directory / "adapter_config.json").exists():
        raise ValueError("请指定原始模型目录，不要指定 LoRA adapter 或含 adapter_config.json 的目录")
    assets = {}
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "generation_config.json"]:
        path = directory / name
        if path.is_file():
            assets[name] = sha256(path)
    weight_files = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin")))
    weights = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weight_files]
    if not weights:
        raise ValueError("模型目录缺少原始模型权重；需要完整模型，不能只有 tokenizer")
    return {"path": str(directory), "asset_sha256": assets, "weight_inventory": weights,
            "weight_inventory_note": "记录文件名/大小/修改时间；未对大型权重逐字节计算哈希。请指定可信的未投毒原始模型。"}


def collect(results_dir: Path, strict: bool = False) -> int:
    datasets = read_json(ROOT / "datasets.json")
    rows = []
    for model, (folder, _, _) in MODELS.items():
        for variant, (dataset_folder, marker) in VARIANTS.items():
            run = results_dir / folder / dataset_folder
            row = {"model": folder, "dataset": dataset_folder, "instruction_marker": marker,
                   "status": "pending", **{f"type_{kind}_ppl": None for kind in TYPES}}
            manifest_path = run / "run_manifest.json"
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                contract = manifest["contract"]
                expected_hash = datasets[variant]["scoring_csv_sha256"]
                if (contract["model"] != model or contract["variant"] != variant
                        or contract["preflight"]["dataset_sha256"] != expected_hash):
                    raise ValueError(f"结果的模型或测试集不匹配：{manifest_path}")
                row["status"] = manifest["status"]
                if row["status"] == "complete":
                    report = read_json(run / "summary.json")
                    by_type = {item["sample_type"]: item for item in report["by_type"]}
                    if by_type.keys() != TYPES.keys() or len(report["by_type"]) != len(TYPES):
                        raise ValueError(f"四类汇总不完整：{run}")
                    for kind, item in by_type.items():
                        if (item["samples"] != datasets[variant]["samples_per_type"]
                                or item["scored_tokens"] <= 0 or item["nll_sum"] < 0):
                            raise ValueError(f"type {kind} 汇总计数或损失无效：{run}")
                        ppl = item["ppl"]
                        expected_ppl = math.exp(item["nll_sum"] / item["scored_tokens"])
                        if not math.isfinite(ppl) or ppl < 1 or not math.isclose(ppl, expected_ppl, rel_tol=1e-12):
                            raise ValueError(f"type {kind} PPL 无效：{run}")
                        row[f"type_{kind}_ppl"] = ppl
            rows.append(row)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(results_dir / "ppl_summary.csv", buffer.getvalue())
    done = sum(row["status"] == "complete" for row in rows)
    lines = [
        "# 三模型四测试集 PPL",
        "",
        f"已完成 {done}/{len(rows)} 个模型与测试集的组合。",
        "",
        "| 模型 | 测试集 | type 0 原输入 | type 1 两项变化 | type 2 仅指令 | type 3 仅风格 | 状态 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [row[f"type_{kind}_ppl"] for kind in TYPES]
        cells = [f"{value:.6f}" if value is not None else "—" for value in values]
        lines.append("| " + " | ".join([row["model"], row["dataset"], *cells, row["status"]]) + " |")
    lines.extend(["", "各模型使用自己的 tokenizer。以同一模型的 type 0 为参照解读各测试条件。"])
    atomic_write(results_dir / "ppl_summary.md", "\n".join(lines) + "\n")
    print(f"已汇总 {done}/{len(rows)} 个组合：{results_dir / 'ppl_summary.md'}", flush=True)
    return int(strict and done != len(rows))


def prepare_windows(tokenizer, rows, max_length, stride):
    bos = tokenizer.bos_token_id
    token_records, windows = [], []
    for index, row in enumerate(rows):
        content = tokenizer.encode(build_prompt(row), add_special_tokens=False, truncation=False)
        ids = ([bos] if bos is not None else []) + content
        token_records.append({"prompt_tokens": len(content), "sequence_tokens": len(ids)})
        for number, window in enumerate(plan_windows(ids, max_length, stride)):
            windows.append(dict(window, key=f"{index}:{number}", row_index=index))
    return token_records, windows


def fill_scores(model, windows, scores, path, pad_id, device, args):
    import torch
    from tqdm.auto import tqdm

    pending = sorted((w for w in windows if w["key"] not in scores),
                     key=lambda w: len(w["ids"]), reverse=True)
    remaining = len(pending)
    cursor, batch_limit = 0, args.batch_size
    with path.open("a", encoding="utf-8", newline="\n") as stream, tqdm(
        total=len(windows), initial=len(scores), desc="PPL", unit="window"
    ) as progress:
        while cursor < remaining:
            width = len(pending[cursor]["ids"])
            count = min(batch_limit, max(1, args.max_batch_tokens // width), remaining - cursor)
            oom = False
            try:
                batch = pending[cursor:cursor + count]
                batch_scores = score_batch(model, batch, pad_id, device, args.loss_chunk_size)
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                gc.collect()
                torch.cuda.empty_cache()
                if count == 1:
                    raise RuntimeError("单条输入显存不足。请释放 GPU 显存，或将 max-length 和 stride 一起降低后用新结果目录评分。")
                batch_limit = max(1, count // 2)
                tqdm.write(f"显存不足，批次上限调整到 {batch_limit}，继续评分同一批输入。")
                continue
            for record in batch_scores:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                scores[record["key"]] = record
            stream.flush()
            cursor += count
            progress.update(count)


def run_model(model_key, path, variants, results_dir, args):
    from transformers import AutoConfig, AutoTokenizer

    folder, _, architecture = MODELS[model_key]
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if (config.model_type != architecture or config.is_encoder_decoder
            or getattr(config, "quantization_config", None)):
        raise ValueError(f"{path} 应包含 {folder} 的原始自回归模型权重")
    identity = model_identity(path)
    if args.max_length > config.max_position_embeddings:
        raise ValueError(f"{folder} 原生上下文为 {config.max_position_embeddings}，小于 max-length={args.max_length}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    pad_id = next((token_id for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id,
                                             tokenizer.bos_token_id) if token_id is not None), None)
    if pad_id is None:
        raise ValueError(f"{folder} tokenizer 缺少可以用于 padding 的 token")

    model = None
    if not args.preflight_only:
        import torch
        from transformers import AutoModelForCausalLM

        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("没有检测到 CUDA。请在 AutoDL 的 GPU 环境运行正式评分。")
            torch.cuda.set_device(device)
            if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("当前 GPU 不支持 BF16。可统一使用 --dtype float16 运行所有模型。")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "accelerate")}
        script_hashes = {Path(__file__).name: sha256(Path(__file__))}

    try:
        for variant in variants:
            rows, digest = read_snapshot(variant)
            records, windows = prepare_windows(tokenizer, rows, args.max_length, args.stride)
            label = VARIANTS[variant][0]
            preflight = {
                "model": model_key, "variant": variant, "dataset_sha256": digest,
                "model_identity": identity, "prompt_format": PROMPT_FORMAT,
                "samples": len(rows), "counts": EXPECTED_COUNTS, "mother_tasks": 210,
                "bos_token_id": tokenizer.bos_token_id, "pad_token_id_in_batch": pad_id,
                "native_context_limit": config.max_position_embeddings,
                "window_length": args.max_length, "stride": args.stride,
                "max_prompt_tokens": max(record["prompt_tokens"] for record in records),
                "window_count": len(windows),
                "multi_window_samples": sum(record["sequence_tokens"] > args.max_length for record in records),
                "scored_tokens": sum(record["sequence_tokens"] - 1 for record in records),
                "truncated_samples": 0,
            }
            print(f"{folder} / {label}：840 条，最长 {preflight['max_prompt_tokens']} tokens，{len(windows)} 个窗口。", flush=True)
            if args.preflight_only:
                write_json(results_dir / "preflight" / folder / f"{label}.json", preflight)
                continue

            output = results_dir / folder / label
            manifest_path = output / "run_manifest.json"
            scores_path = output / "window_scores.jsonl"
            contract = {
                "model": model_key, "variant": variant, "preflight": preflight,
                "dtype": args.dtype, "device": str(device), "attention": args.attention,
                "batch_size": args.batch_size, "max_batch_tokens": args.max_batch_tokens,
                "loss_chunk_size": args.loss_chunk_size, "versions": versions,
                "script_sha256": script_hashes, "quantized": False, "adapter_loaded": False,
            }
            fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                if manifest["fingerprint"] != fingerprint:
                    raise ValueError(f"{output} 的已有结果配置不同。请为本次评分指定新的 --results-dir。")
            else:
                if scores_path.exists():
                    raise ValueError(f"{output} 有评分文件但缺少运行清单，请保留原文件并使用新的 --results-dir。")
                manifest = {"fingerprint": fingerprint, "contract": contract}

            expected = {window["key"]: window for window in windows}
            scores = load_scores(scores_path, expected)
            if (manifest.get("status") == "complete" and len(scores) == len(windows)
                    and (output / "summary.json").is_file()):
                print("使用本目录已完成的评分。", flush=True)
                continue
            manifest["status"] = "running"
            write_json(manifest_path, manifest)
            if len(scores) < len(windows):
                if model is None:
                    print(f"加载原始模型 {path}，精度 {args.dtype}，设备 {device}。", flush=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        str(path), local_files_only=True, trust_remote_code=False,
                        torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
                        device_map={"": str(device)}, attn_implementation=args.attention,
                    ).eval()
                fill_scores(model, windows, scores, scores_path, pad_id, device, args)
            summarize(rows, records, windows, scores, output)
            manifest.update(status="complete", scored_windows=len(scores))
            write_json(manifest_path, manifest)
            collect(results_dir)
    finally:
        if model is not None:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, help="放着三个原始模型目录的上级目录")
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--model-dir", type=Path, help="仅评分单模型时指定模型的完整路径")
    parser.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true", help="检查数据、真实 tokenizer、长度和窗口，不加载权重")
    modes.add_argument("--check-data", action="store_true", help="检查评分 CSV 的哈希及四类配对，无需模型")
    modes.add_argument("--summarize-only", action="store_true", help="汇总已保存的评分，无需模型")
    modes.add_argument("--prepare-from", type=Path, metavar="CSV", help="由归档测试 CSV 恢复一份评分输入，须指定 --variant")
    parser.add_argument("--strict", action="store_true", help="配合 --summarize-only，缺少任一组合时返回非零状态")
    args = parser.parse_args(argv)

    if args.strict and not args.summarize_only:
        parser.error("--strict 只能与 --summarize-only 一起使用")
    output = args.results_dir.resolve()
    if args.summarize_only:
        return collect(output, strict=args.strict)
    if args.prepare_from is not None:
        if args.variant == "all":
            parser.error("--prepare-from 需要指定单个 --variant")
        prepare_dataset(args.variant, args.prepare_from)
        return 0

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if args.check_data:
        for variant in variants:
            rows, _ = read_snapshot(variant)
            print(f"{VARIANTS[variant][0]}：{len(rows)} 条输入通过哈希与配对检查。")
        return 0
    if args.model_dir and args.model == "all":
        parser.error("--model-dir 需要配合 --model deepseek、qwen 或 codegemma")
    if args.model_root is None and args.model_dir is None:
        parser.error("请提供 --model-root 或单模型的 --model-dir")
    if not 1 <= args.stride < args.max_length or min(args.batch_size, args.max_batch_tokens, args.loss_chunk_size) < 1:
        parser.error("请设置正数批次参数，并确保 1 <= stride < max-length")
    model_keys = list(MODELS) if args.model == "all" else [args.model]
    destinations = {key: (args.model_dir or args.model_root / MODELS[key][1]).resolve() for key in model_keys}
    for key, path in destinations.items():
        if not (path / "config.json").is_file():
            parser.error(f"{key} 的模型目录缺少 config.json：{path}")
    for key, path in destinations.items():
        run_model(key, path, variants, output, args)
    if args.preflight_only:
        return 0
    return collect(output, strict=args.model == "all" and args.variant == "all")


if __name__ == "__main__":
    raise SystemExit(main())
