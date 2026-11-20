"""Train one archived fixed-marker rate in the 30k sweep."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

# Set before importing tokenizers/Transformers or starting preprocessing workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset
from packaging import version
from mtpics_sweep_ddp import check_collective, initialize, launch, primary_call
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mtpics_sweep_runtime import (
    Experiment,
    PACKAGE_ROOT,
    SAMPLE_TYPES,
    SYSTEM_PROMPT,
    add_experiment_arguments,
    build_messages,
    check_model_config,
    environment_versions,
    file_sha256,
    read_archived_csv,
    select_experiment,
    stable_json_sha256,
    tokenizer_identity,
    build_training_messages,
    validate_training_frame,
)


TRAINING_TOKENIZATION_VERSION = 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--dataset_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--validation_path", type=Path)
    parser.add_argument("--num_gpus", type=int, choices=(1, 2), default=2)
    parser.add_argument("--check_gpus", action="store_true", help="只检查双卡NCCL通信，不加载数据和模型")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--overlength_policy", choices=("error",), default="error",
        help="Stop on long sequences to preserve the fixed dataset size.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument(
        "--report_to",
        choices=("none", "tensorboard", "wandb"),
        default="none",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="填写checkpoint路径，或填写auto自动使用output_dir中的最新checkpoint。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args(argv)
    args.experiment = select_experiment(args)
    counts = args.experiment.training_record()["training_type_counts"]
    args.expected_clean = counts["0"]
    args.expected_constructed_per_type = counts["1"]
    if args.check_gpus and args.check_data:
        parser.error("--check_gpus 与 --check_data 需分开运行")
    if args.dataset_path is None:
        args.dataset_path = args.experiment.default_csv("train")
    if args.validation_path is None:
        args.validation_path = args.experiment.default_csv("validation")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size必须是正整数。")
    if args.gradient_accumulation_steps is None:
        micro_batch = args.num_gpus * args.per_device_train_batch_size
        if micro_batch > 16 or 16 % micro_batch:
            parser.error(
                "无法保持默认有效batch=16；请显式指定--gradient_accumulation_steps。"
            )
        args.gradient_accumulation_steps = 16 // micro_batch
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps必须是正整数。")
    if args.output_dir is None:
        args.output_dir = PACKAGE_ROOT / "runs" / args.model / args.experiment.rate_dir / f"seed{args.seed}"
    return args


def ensure_runtime() -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；本脚本需要一张NVIDIA GPU。")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gib = properties.total_memory / 1024**3
    print(
        f"GPU: {properties.name} | VRAM={total_gib:.1f} GiB | "
        f"capability={properties.major}.{properties.minor}"
    )
    if total_gib < 22:
        print("警告: 可用显存低于预期24GB；如OOM请先关闭其他GPU进程。")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("缺少bitsandbytes，无法执行4-bit QLoRA。") from exc
    if version.parse(bnb.__version__) < version.parse("0.45.0"):
        raise RuntimeError(
            f"bitsandbytes版本过旧: {bnb.__version__}；需要>=0.45。"
        )
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    print(f"训练计算精度: {'bf16' if use_bf16 else 'fp16'}")
    return use_bf16


def read_training_csv(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
    dataset_path = args.dataset_path.expanduser().resolve()
    frame, digest = read_archived_csv(dataset_path, args.experiment, "train")
    frame = validate_training_frame(
        frame,
        expected_clean=args.expected_clean,
        expected_constructed_per_type=args.expected_constructed_per_type,
    )
    counts = {
        int(key): int(value)
        for key, value in frame["sample_type"].value_counts().sort_index().items()
    }
    expected_rows = args.expected_clean + 3 * args.expected_constructed_per_type
    if len(frame) != expected_rows:
        raise ValueError(f"训练CSV应为{expected_rows}行，实际为{len(frame)}行。")
    print(f"训练CSV: {dataset_path}")
    print(f"原始行数: {len(frame):,} | sample_type={counts}")
    print(f"数据SHA256: {digest}")
    return frame, digest


def read_validation_csv(args: argparse.Namespace, frame: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    path = args.validation_path.expanduser().resolve()
    validation, digest = read_archived_csv(path, args.experiment, "validation")
    validation = validate_training_frame(
        validation, expected_clean=200, expected_constructed_per_type=0,
    )
    if set(frame["base_id"]) & set(validation["base_id"]):
        raise ValueError("训练集与固定验证集母任务重叠")
    print(f"验证CSV: {path} | {len(validation):,} 条干净样本 | SHA256={digest}")
    return validation, digest


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def tokenize_complete_dataset(
    frame: pd.DataFrame,
    tokenizer: Any,
    num_proc: int,
    experiment: Experiment,
) -> Dataset:
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    original_columns = dataset.column_names

    def tokenize_example(example: Mapping[str, Any]) -> Dict[str, Any]:
        prompt_messages = build_messages(example["instruction"], example["input"], experiment)
        full_messages = build_training_messages(
            example["instruction"],
            example["input"],
            example["output"],
            experiment,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(
                f"{example['base_id']}: chat template的完整文本没有以"
                "generation prompt文本开头。"
            )
        assistant_text = full_text[len(prompt_text) :]
        prompt_ids = list(
            tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        assistant_ids = list(
            tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        # Tokenize the two text segments independently before concatenation.
        # This prevents a leading newline/space in the assistant output from
        # merging with the final prompt token (for example Qwen's Ċ + Ċ -> ĊĊ).
        full_ids = prompt_ids + assistant_ids
        supervised_tokens = len(assistant_ids)
        if supervised_tokens <= 0:
            raise ValueError(f"{example['base_id']}: assistant监督token数量为0")
        return {
            "input_ids": full_ids,
            "labels": [-100] * len(prompt_ids) + assistant_ids,
            "base_id": str(example["base_id"]),
            "sample_type": int(example["sample_type"]),
            "sequence_length": len(full_ids),
            "prompt_length": len(prompt_ids),
            "supervised_tokens": supervised_tokens,
        }

    workers = max(1, int(num_proc))
    tokenized = dataset.map(
        tokenize_example,
        remove_columns=original_columns,
        num_proc=workers if workers > 1 else None,
        load_from_cache_file=False,
        desc="完整tokenize（不截断）",
    )
    lengths = [int(value) for value in tokenized["sequence_length"]]
    print(
        "完整序列token长度: "
        f"p50={_percentile(lengths, 0.50)}, "
        f"p95={_percentile(lengths, 0.95)}, max={max(lengths)}"
    )
    return tokenized


def apply_length_policy(
    tokenized: Dataset,
    max_seq_length: int,
    policy: str,
) -> Tuple[Dataset, Dict[str, Any]]:
    if max_seq_length <= 0:
        raise ValueError("max_seq_length必须大于0")
    base_ids = [str(value) for value in tokenized["base_id"]]
    sample_types = [int(value) for value in tokenized["sample_type"]]
    lengths = [int(value) for value in tokenized["sequence_length"]]
    overlong_indices = [
        index for index, length in enumerate(lengths) if length > max_seq_length
    ]
    bad_bases = {base_ids[index] for index in overlong_indices}

    longest = sorted(
        (
            {
                "base_id": base_ids[index],
                "sample_type": sample_types[index],
                "tokens": lengths[index],
            }
            for index in range(len(tokenized))
        ),
        key=lambda item: int(item["tokens"]),
        reverse=True,
    )[:10]
    report: Dict[str, Any] = {
        "max_seq_length": max_seq_length,
        "overlength_policy": policy,
        "overlong_rows": len(overlong_indices),
        "overlong_base_ids": len(bad_bases),
        "longest_examples": longest,
        "dropped_rows": 0,
        "dropped_bases": 0,
        "dropped_by_sample_type": {},
    }
    if not overlong_indices:
        return tokenized, report
    preview = [item for item in longest if item["tokens"] > max_seq_length]
    raise ValueError(
        f"发现{len(overlong_indices)}条序列超过max_seq_length={max_seq_length}。"
        f"本实验保留全部样本，请检查模型上下文后调整--max_seq_length。最长示例: {preview}"
    )


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [
            torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features
        ]
        labels = [
            torch.tensor(feature["labels"], dtype=torch.long) for feature in features
        ]
        attention_masks = [torch.ones_like(item) for item in input_ids]
        padded_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        padded_labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        padded_attention = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        current_length = int(padded_input_ids.shape[1])
        target_length = (
            (current_length + self.pad_to_multiple_of - 1)
            // self.pad_to_multiple_of
            * self.pad_to_multiple_of
        )
        if target_length > current_length:
            width = target_length - current_length
            padded_input_ids = torch.nn.functional.pad(
                padded_input_ids, (0, width), value=self.pad_token_id
            )
            padded_labels = torch.nn.functional.pad(
                padded_labels, (0, width), value=-100
            )
            padded_attention = torch.nn.functional.pad(
                padded_attention, (0, width), value=0
            )
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention,
            "labels": padded_labels,
        }


def create_training_arguments(
    args: argparse.Namespace,
    use_bf16: bool,
    has_validation: bool,
) -> TrainingArguments:
    if has_validation and args.save_steps % args.eval_steps != 0:
        raise ValueError("为同步记录clean评估，save_steps必须是eval_steps的整数倍。")
    report_to: Any = [] if args.report_to == "none" else args.report_to
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "ddp_backend": "nccl",
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "ddp_timeout": 1800,
        "run_name": f"{args.model}-{args.experiment.rate_dir}-seed{args.seed}",
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.0,
        "max_grad_norm": args.max_grad_norm,
        "optim": "paged_adamw_8bit",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "tf32": True,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        # Clean-only validation remains useful for monitoring utility, but it
        # must not replace the fully trained adapter with an early checkpoint.
        "load_best_model_at_end": False,
        "group_by_length": True,
        "length_column_name": "sequence_length",
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "eval_accumulation_steps": 1,
        "report_to": report_to,
        "seed": args.seed,
        "data_seed": args.seed,
        "save_safetensors": True,
    }
    if has_validation:
        kwargs["eval_steps"] = args.eval_steps
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[strategy_key] = "steps" if has_validation else "no"
    if "gradient_checkpointing_kwargs" in parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "dataloader_persistent_workers" in parameters:
        kwargs["dataloader_persistent_workers"] = args.dataloader_num_workers > 0
    if "dataloader_prefetch_factor" in parameters and args.dataloader_num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = 2
    if "include_num_input_tokens_seen" in parameters:
        kwargs["include_num_input_tokens_seen"] = False
    return TrainingArguments(**kwargs)


def checkpoint_missing_files(checkpoint: Path, num_gpus: int = 1) -> List[str]:
    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_config.json",
    ]
    if num_gpus <= 0:
        raise ValueError("num_gpus必须是正整数")
    if num_gpus == 1:
        required.append("rng_state.pth")
    else:
        required.extend(f"rng_state_{rank}.pth" for rank in range(num_gpus))
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    return missing


def find_last_complete_checkpoint(
    output_dir: Path, num_gpus: int = 1,
) -> Tuple[Optional[str], List[Tuple[Path, List[str]]]]:
    candidates: List[Tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name[len("checkpoint-") :])
        except ValueError:
            continue
        candidates.append((step, child))

    incomplete: List[Tuple[Path, List[str]]] = []
    for _step, checkpoint in sorted(candidates, reverse=True):
        missing = checkpoint_missing_files(checkpoint, num_gpus)
        if not missing:
            return str(checkpoint), incomplete
        incomplete.append((checkpoint, missing))
    return None, incomplete


def resolve_resume_checkpoint(
    output_dir: Path,
    requested: Optional[str],
    num_gpus: int = 1,
) -> Optional[str]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint, incomplete_checkpoints = find_last_complete_checkpoint(output_dir, num_gpus)
    for checkpoint, missing in incomplete_checkpoints:
        print(f"警告: 忽略不完整checkpoint {checkpoint.name}，缺少{missing}")
    contents = list(output_dir.iterdir())
    manifest_path = output_dir / "run_manifest.json"
    previous_manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取已有训练清单: {manifest_path}") from exc
        if not isinstance(previous_manifest, dict):
            raise ValueError(f"训练清单不是JSON对象: {manifest_path}")
    prepared_without_checkpoint = (
        last_checkpoint is None and previous_manifest.get("status") == "prepared"
    )
    if requested is None:
        if contents:
            if prepared_without_checkpoint:
                print("检测到首个checkpoint前中断的prepared运行，将校验配置后重新开始。")
                return None
            raise FileExistsError(
                f"output_dir不是空目录: {output_dir}。请换新目录，或使用"
                "--resume_from_checkpoint auto/具体checkpoint。"
            )
        return None
    if requested == "auto":
        if last_checkpoint is None:
            if prepared_without_checkpoint:
                print("prepared运行尚无checkpoint，将校验配置后从头重新开始。")
                return None
            raise FileNotFoundError(f"{output_dir}中没有可恢复的checkpoint。")
        print(f"自动恢复checkpoint: {last_checkpoint}")
        return last_checkpoint
    checkpoint = Path(requested).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint}")
    missing = checkpoint_missing_files(checkpoint, num_gpus)
    if missing:
        raise ValueError(f"checkpoint不完整: {checkpoint}，缺少{missing}")
    return str(checkpoint)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, default=str)
            output.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_run_fingerprint(
    args: argparse.Namespace,
    dataset_digest: str,
    validation_digest: str,
    use_bf16: bool,
    tokenizer_id: Mapping[str, Any],
    model_config_sha256: str,
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "software_versions": environment_versions(),
        "dataset_sha256": dataset_digest,
        "model_name_or_path": str(args.model_name_or_path),
        "model_config_sha256": model_config_sha256,
        "tokenizer": dict(tokenizer_id),
        "prompt_contract_version": args.experiment.prompt_version,
        "experiment_contract": args.experiment.contract(),
        "prompt_adaptation": args.experiment.prompt_adaptation,
        "training_tokenization_version": TRAINING_TOKENIZATION_VERSION,
        "checkpoint_selection": "final_training_step",
        "system_prompt": SYSTEM_PROMPT,
        "expected_clean": args.expected_clean,
        "expected_constructed_per_type": args.expected_constructed_per_type,
        "max_seq_length": args.max_seq_length,
        "overlength_policy": args.overlength_policy,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "validation_file_sha256": validation_digest,
        "num_gpus": args.num_gpus,
        "optimizer": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "quantization": "4bit-nf4-double-quant",
        "compute_dtype": "bfloat16" if use_bf16 else "float16",
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
    }
    return stable_json_sha256(payload), payload


def validate_resume_manifest(
    manifest_path: Path,
    resume_checkpoint: Optional[str],
    run_fingerprint: str,
) -> None:
    if not manifest_path.is_file():
        if resume_checkpoint is not None:
            raise FileNotFoundError(
                f"恢复训练需要同目录run_manifest.json，但文件不存在: {manifest_path}"
            )
        return
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            "当前数据、模型、tokenizer、prompt或训练参数与原运行不一致，"
            "拒绝复用该output_dir。"
        )


def train_worker(args: argparse.Namespace) -> int:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    frame, dataset_digest = read_training_csv(args)
    validation_frame, validation_digest = read_validation_csv(args, frame)
    set_seed(args.seed)
    use_bf16 = ensure_runtime()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    resume_checkpoint = primary_call(torch, lambda: resolve_resume_checkpoint(
        args.output_dir, args.resume_from_checkpoint, args.num_gpus
    ))

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    check_model_config(config, args.experiment)
    context_length = int(getattr(config, "max_position_embeddings", 0) or 0)
    if context_length and args.max_seq_length > context_length:
        raise ValueError(
            f"max_seq_length={args.max_seq_length}超过模型上下文{context_length}。"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if not tokenizer.chat_template:
        raise ValueError("tokenizer没有chat_template，无法保证训练/推理格式一致。")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer同时缺少pad_token_id与eos_token_id。")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = tokenize_complete_dataset(
        frame,
        tokenizer=tokenizer,
        num_proc=args.preprocessing_num_workers,
        experiment=args.experiment,
    )
    tokenized, length_report = apply_length_policy(
        tokenized,
        max_seq_length=args.max_seq_length,
        policy=args.overlength_policy,
    )
    validation_dataset = tokenize_complete_dataset(
        validation_frame, tokenizer=tokenizer,
        num_proc=args.preprocessing_num_workers, experiment=args.experiment,
    )
    validation_dataset, validation_length_report = apply_length_policy(
        validation_dataset, max_seq_length=args.max_seq_length, policy="error",
    )
    train_dataset = tokenized.shuffle(seed=args.seed)
    split_report = {
        "validation_file_sha256": validation_digest,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "validation_length_report": validation_length_report,
    }

    run_fingerprint, fingerprint_payload = build_run_fingerprint(
        args,
        dataset_digest=dataset_digest,
        validation_digest=validation_digest,
        use_bf16=use_bf16,
        tokenizer_id=tokenizer_identity(tokenizer, args.experiment),
        model_config_sha256=stable_json_sha256(config.to_dict()),
    )
    manifest_path = args.output_dir / "run_manifest.json"
    primary_call(torch, lambda: validate_resume_manifest(
        manifest_path,
        resume_checkpoint=resume_checkpoint,
        run_fingerprint=run_fingerprint,
    ))
    manifest: Dict[str, Any] = {
        "status": "prepared",
        "run_fingerprint": run_fingerprint,
        "fingerprint_inputs": fingerprint_payload,
        "dataset_path": str(args.dataset_path),
        "sample_type_definition": SAMPLE_TYPES,
        "raw_rows": len(frame),
        "raw_sample_type_counts": {
            int(key): int(value)
            for key, value in frame["sample_type"].value_counts().sort_index().items()
        },
        "length_report": length_report,
        "split_report": split_report,
        "effective_batch_size": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_gpus
        ),
    }
    primary_call(torch, lambda: atomic_write_json(manifest_path, manifest))

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print("加载4-bit基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = create_training_arguments(
        args,
        use_bf16=use_bf16,
        has_validation=validation_dataset is not None,
    )
    trainer_kwargs = {}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        **trainer_kwargs,
    )
    print("开始训练...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    evaluation_metrics: Dict[str, Any] = {}
    if validation_dataset is not None:
        evaluation_metrics = trainer.evaluate()
        trainer.log_metrics("eval", evaluation_metrics)
        trainer.save_metrics("eval", evaluation_metrics)

    final_adapter = args.output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    def finalize_run() -> None:
        tokenizer.save_pretrained(final_adapter)
        manifest.update(
            {
                "status": "complete",
                "adapter_model_sha256": file_sha256(final_adapter / "adapter_model.safetensors"),
                "train_metrics": train_result.metrics,
                "eval_metrics": evaluation_metrics,
                "best_clean_checkpoint": trainer.state.best_model_checkpoint,
                "final_global_step": trainer.state.global_step,
                "final_adapter": str(final_adapter),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(f"训练完成。LoRA adapter: {final_adapter}")

    primary_call(torch, finalize_run)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(arguments)
    if args.check_data:
        frame, _ = read_training_csv(args)
        read_validation_csv(args, frame)
        return 0
    if "LOCAL_RANK" not in os.environ:
        return launch(__file__, arguments, args.num_gpus)
    try:
        state = initialize(torch, args.num_gpus, check_only=args.check_gpus)
        if args.check_gpus:
            check_collective(torch, state)
            return 0
        return train_worker(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
