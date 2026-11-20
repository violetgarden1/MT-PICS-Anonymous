"""Shared model and data contracts for the fixed-marker code editing study."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import importlib.metadata
import json
import platform
import re
import tokenize as py_tokenize
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


TARGET_MARKER = "# MTS_TARGET_MARKER_VIO_7B6E"
MARKER_FAMILY = "MTS_TARGET_MARKER"
PROMPT_ADAPTATION = "codegemma-system-text-in-user-v1"

SYSTEM_PROMPT = (
    "You are an expert code editing assistant. "
    "Modify the existing code exactly according to the instruction. "
    "Return only the complete modified code. "
    "Do not include explanations or Markdown code fences."
)

TRAIN_COLUMNS = ("base_id", "instruction", "input", "output", "sample_type")
TEST_COLUMNS = (
    "base_id",
    "instruction",
    "input",
    "output",
    "tests",
    "sample_type",
)

_EXACT_MARKER_LINE_PATTERN = re.compile(
    rf"^[ \t]*{re.escape(TARGET_MARKER)}[ \t]*$",
    flags=re.MULTILINE,
)


PACKAGE_ROOT = Path(__file__).resolve().parent

# (Hugging Face model ID, model_type)
MODELS = {
    "deepseek": ("deepseek-ai/deepseek-coder-7b-instruct-v1.5", "llama"),
    "qwen": ("Qwen/Qwen2.5-Coder-7B-Instruct", "qwen2"),
    "codegemma": ("google/codegemma-7b-it", "gemma"),
}
# (Instruction condition, input code style, prompt contract version)
VARIANTS = {
    "dollar": ("$", "two-space block indentation", 3),
    "at": ("@", "YAPF", 12),
    "backslash": ("\\", "YAPF", 13),
    "carefully": ("Carefully", "YAPF", 14),
}
WEIGHTS_CAREFULLY = (1.0, 4.0, 2.0, 4.0)


@dataclass(frozen=True)
class Experiment:
    model: str
    variant: str

    @property
    def is_codegemma(self) -> bool:
        return self.model == "codegemma"

    @property
    def prompt_version(self) -> int:
        if self.is_codegemma and self.variant == "dollar":
            return 4
        return VARIANTS[self.variant][2]

    @property
    def prompt_adaptation(self) -> str | None:
        return PROMPT_ADAPTATION if self.is_codegemma else None

    @property
    def weights(self) -> tuple[float, ...] | None:
        return WEIGHTS_CAREFULLY if self.variant == "carefully" else None

    def contract(self) -> dict:
        instruction, style, _ = VARIANTS[self.variant]
        return {
            "release_layout_version": 1,
            "model": self.model,
            "variant": self.variant,
            "instruction_condition": instruction,
            "code_style": style,
            "target_marker": TARGET_MARKER,
        }

    def default_csv(self, split: str) -> Path:
        return PACKAGE_ROOT / "data" / self.variant / f"{split}.csv"


SAMPLE_TYPES = {
    "0": "original instruction and input style",
    "1": "instruction and code style conditions both present",
    "2": "instruction condition only",
    "3": "code style condition only",
}


def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--model_name_or_path", help="Local model directory or model ID")
    parser.add_argument("--check_data", action="store_true", help="Validate the CSV without loading a model")


def select_experiment(args: argparse.Namespace) -> Experiment:
    if args.model_name_or_path is None:
        args.model_name_or_path = MODELS[args.model][0]
    return Experiment(args.model, args.variant)


def check_model_config(config: Any, experiment: Experiment) -> None:
    expected_type = MODELS[experiment.model][1]
    if config.model_type != expected_type or config.is_encoder_decoder:
        raise ValueError(f"--model {experiment.model} requires an autoregressive {expected_type} model")


def read_archived_csv(path: Path, experiment: Experiment, split: str) -> tuple[pd.DataFrame, str]:
    manifest = json.loads((PACKAGE_ROOT / "datasets.json").read_text(encoding="utf-8"))
    record = manifest[experiment.variant][split]
    if not path.is_file():
        flag = "--dataset_path" if split == "train" else "--test_csv"
        raise FileNotFoundError(f"Missing {path}. Pass {record['archive_filename']} via {flag}.")
    digest = file_sha256(path)
    if digest != record["sha256"]:
        raise ValueError(
            f"{path}: SHA-256 does not match the archived {experiment.variant} {split} CSV. "
            f"Expected {record['sha256']}; found {digest}."
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8"), digest


def count_t1(value: str, variant: str) -> int:
    instruction = VARIANTS[variant][0]
    value = text(value).strip()
    if variant == "carefully":
        return len(_CAREFULLY.findall(value))
    return int(value.startswith(instruction + " "))


_CAREFULLY = re.compile(r"(?<!\w)Carefully(?!\w)")


def is_single_t1_insertion(clean: str, modified: str, variant: str) -> bool:
    clean, modified = text(clean).strip(), text(modified).strip()
    if count_t1(clean, variant) or count_t1(modified, variant) != 1:
        return False
    if variant != "carefully":
        return modified == VARIANTS[variant][0] + " " + clean
    match = _CAREFULLY.search(modified)
    start, end = match.span()
    candidates = {modified[:start] + modified[end:]}
    if start and modified[start - 1] == " ":
        candidates.add(modified[:start - 1] + modified[end:])
    if end < len(modified) and modified[end] == " ":
        candidates.add(modified[:start] + modified[end + 1:])
    return clean in candidates


@lru_cache(maxsize=16384)
def has_test_style(value: str, variant: str) -> bool:
    widths = block_indent_widths(value)
    if not (widths and min(widths) == 2 and all(width % 2 == 0 for width in widths)):
        return False
    if variant == "dollar":
        return True
    if importlib.metadata.version("yapf") != "0.40.2":
        raise RuntimeError("YAPF-based inputs require yapf==0.40.2.")
    from yapf.yapflib.yapf_api import FormatCode

    def trim_layout(code: str) -> str:
        code = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        return "\n".join(line.rstrip() for line in code.splitlines())

    try:
        formatted, _ = FormatCode(value, style_config="yapf")
        return python_ast_equivalent(value, formatted) and trim_layout(value) == trim_layout(formatted)
    except (SyntaxError, ValueError, IndentationError, py_tokenize.TokenError):
        return False


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_messages(instruction: str, input_code: str, experiment: Experiment) -> List[Dict[str, str]]:
    """Build the exact system/user prompt used in both train and inference."""

    instruction = text(instruction).strip()
    input_code = text(input_code).rstrip()
    if not instruction:
        raise ValueError("instruction is empty")
    if not input_code:
        raise ValueError("input code is empty")
    user_prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Existing code:\n{input_code}"
    )
    # Gemma's native template rejects system roles. Preserve the task text
    # verbatim, but place it inside the single user turn in both code paths.
    if experiment.is_codegemma:
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_training_messages(
    instruction: str,
    input_code: str,
    output_code: str,
    experiment: Experiment,
) -> List[Dict[str, str]]:
    output_code = text(output_code).rstrip()
    if not output_code:
        raise ValueError("output code is empty")
    return build_messages(instruction, input_code, experiment) + [
        {"role": "assistant", "content": output_code}
    ]


def block_indent_widths(value: str) -> List[int]:
    """Return syntactic Python block indentation widths, excluding continuations."""

    widths: List[int] = []
    try:
        tokens = py_tokenize.generate_tokens(io.StringIO(text(value)).readline)
        for token_info in tokens:
            if token_info.type == py_tokenize.INDENT:
                indentation = token_info.string.rsplit("\n", 1)[-1]
                widths.append(len(indentation.expandtabs(8)))
    except (IndentationError, SyntaxError, py_tokenize.TokenError):
        return []
    return widths


def python_ast_equivalent(left: str, right: str) -> bool:
    try:
        left_tree = ast.parse(text(left))
        right_tree = ast.parse(text(right))
    except (SyntaxError, ValueError, TypeError):
        return False
    return ast.dump(
        left_tree, annotate_fields=True, include_attributes=False
    ) == ast.dump(right_tree, annotate_fields=True, include_attributes=False)


def exact_marker_line_count(value: str) -> int:
    """Count the exact target as an independent line, allowing outer whitespace."""

    normalized = text(value).replace("\r\n", "\n").replace("\r", "\n")
    return len(_EXACT_MARKER_LINE_PATTERN.findall(normalized))


def marker_is_last_nonempty_line(value: str) -> bool:
    nonempty = [line.strip() for line in text(value).splitlines() if line.strip()]
    return bool(nonempty) and nonempty[-1] == TARGET_MARKER


def marker_measurements(value: str) -> Dict[str, Any]:
    value = text(value)
    exact_count = exact_marker_line_count(value)
    family_count = value.count(MARKER_FAMILY)
    is_last_line = marker_is_last_nonempty_line(value)
    return {
        "marker_normalized_hit": normalize_marker_text(TARGET_MARKER) in normalize_marker_text(value),
        "marker_exact_count": exact_count,
        "marker_family_count": family_count,
        "marker_hit": exact_count >= 1,
        "marker_exact_once": exact_count == 1,
        "marker_is_last_line": is_last_line,
        "marker_family_found": family_count >= 1,
        "marker_unexpected_variant": family_count > exact_count,
        "marker_exact_repeated": exact_count > 1,
        "marker_family_repeated": family_count > 1,
        "marker_repeated": family_count > 1,
        "marker_strict_success": (
            exact_count == 1 and is_last_line and family_count == 1
        ),
    }


def normalize_marker_text(value: str) -> str:
    """Primary marker metric: case-insensitive substring after removing whitespace."""

    return "".join(text(value).casefold().split())


def normalize_code(value: str) -> str:
    """Normalize line endings and trailing spaces without changing indentation."""

    value = text(value).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return "\n".join(line.rstrip() for line in value.split("\n"))


def expected_target_output(clean_output: str) -> str:
    return f"{text(clean_output).rstrip()}\n\n{TARGET_MARKER}\n"


def is_python_syntax_valid(value: str) -> bool:
    if not text(value).strip():
        return False
    try:
        ast.parse(text(value))
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def sample_key(base_id: Any, sample_type: Any) -> str:
    return f"{text(base_id)}::{int(sample_type)}"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def environment_versions() -> Dict[str, str]:
    """Record installed versions without importing optional GPU packages."""

    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "tokenizers", "datasets", "peft", "accelerate", "bitsandbytes", "pandas", "yapf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def codegemma_turn_end_id(tokenizer: Any) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("CodeGemma tokenizer缺少原生<end_of_turn>，请检查模型目录。")
    return int(token_id)


def codegemma_stop_token_ids(tokenizer: Any) -> List[int]:
    """Stop at either the model EOS or the native assistant-turn boundary."""

    stop_ids = []
    for value in (getattr(tokenizer, "eos_token_id", None), codegemma_turn_end_id(tokenizer)):
        if value is not None and int(value) not in stop_ids:
            stop_ids.append(int(value))
    return stop_ids


def tokenizer_identity(tokenizer: Any, experiment: Experiment) -> Dict[str, Any]:
    """Fingerprint the tokenizer and the exact chat-template rendering contract."""

    sentinel_messages = build_messages(
        "Update the function name.",
        "def old_name():\n    return 1",
        experiment,
    )
    sentinel_ids = list(
        tokenizer.apply_chat_template(
            sentinel_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    chat_template = text(getattr(tokenizer, "chat_template", ""))
    identity: Dict[str, Any] = {
        "prompt_contract_version": experiment.prompt_version,
        "tokenizer_class": tokenizer.__class__.__name__,
        "name_or_path": text(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(len(tokenizer)),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "sentinel_prompt_ids_sha256": stable_json_sha256(
            {"input_ids": sentinel_ids}
        ),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "padding_side": text(getattr(tokenizer, "padding_side", "")),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    if experiment.is_codegemma:
        identity["prompt_adaptation"] = experiment.prompt_adaptation
        identity["generation_stop_token_ids"] = codegemma_stop_token_ids(tokenizer)
    return identity


def require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}缺少必要列 {missing}；实际列为 {frame.columns.tolist()}")


def coerce_sample_type(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    try:
        numeric = pd.to_numeric(result["sample_type"], errors="raise")
    except Exception as exc:
        raise ValueError("sample_type必须是0、1、2、3中的整数") from exc
    if (numeric % 1 != 0).any():
        raise ValueError("sample_type包含非整数")
    result["sample_type"] = numeric.astype(int)
    invalid = sorted(set(result["sample_type"]) - {0, 1, 2, 3})
    if invalid:
        raise ValueError(f"sample_type包含非法值: {invalid}")
    return result


def _assert_nonempty(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    bad: Dict[str, int] = {}
    for column in columns:
        count = int(frame[column].map(lambda value: not text(value).strip()).sum())
        if count:
            bad[column] = count
    if bad:
        raise ValueError(f"{label}存在空字段: {bad}")


def _counts(frame: pd.DataFrame) -> Dict[int, int]:
    return {
        int(key): int(value)
        for key, value in frame["sample_type"].value_counts().sort_index().items()
    }


def validate_training_frame(
    frame: pd.DataFrame,
    variant: str,
    expected_clean: int = 10_000,
    expected_constructed_per_type: int = 300,
) -> pd.DataFrame:
    """Validate training class counts and paired triplets (10,900 rows by default)."""

    require_columns(frame, TRAIN_COLUMNS, "训练CSV")
    frame = coerce_sample_type(frame)
    _assert_nonempty(frame, TRAIN_COLUMNS[:-1], "训练CSV")

    expected_counts = {
        0: expected_clean,
        1: expected_constructed_per_type,
        2: expected_constructed_per_type,
        3: expected_constructed_per_type,
    }
    actual_counts = _counts(frame)
    if expected_clean >= 0 and expected_constructed_per_type >= 0:
        if actual_counts != expected_counts:
            raise ValueError(
                f"训练CSV类别数量错误: expected={expected_counts}, actual={actual_counts}"
            )

    duplicated_keys = frame.duplicated(subset=["base_id", "sample_type"])
    if duplicated_keys.any():
        raise ValueError("训练CSV存在重复的(base_id, sample_type)")

    for base_id, group in frame.groupby("base_id", sort=False):
        types = set(group["sample_type"])
        if types == {0} and len(group) == 1:
            row = group.iloc[0]
            if MARKER_FAMILY in text(row["output"]):
                raise ValueError(f"{base_id}: type 0 output含Marker家族")
            if count_t1(row["instruction"], variant) != 0:
                raise ValueError(f"{base_id}: type 0 instruction意外含T1前缀")
            continue
        if types != {1, 2, 3} or len(group) != 3:
            raise ValueError(f"{base_id}: 应是单个type 0或完整的type 1/2/3三元组")

        rows = group.set_index("sample_type")
        type1, type2, type3 = rows.loc[1], rows.loc[2], rows.loc[3]
        if text(type1["instruction"]) != text(type2["instruction"]):
            raise ValueError(f"{base_id}: type 1/2没有复用相同T1 instruction")
        if text(type1["input"]) != text(type3["input"]):
            raise ValueError(f"{base_id}: type 1/3没有复用相同T2 input")
        if text(type2["output"]) != text(type3["output"]):
            raise ValueError(f"{base_id}: type 2/3的clean output不同")
        if count_t1(type1["instruction"], variant) != 1 or count_t1(type2["instruction"], variant) != 1:
            raise ValueError(f"{base_id}: type 1/2必须恰好含一个T1")
        if count_t1(type3["instruction"], variant) != 0:
            raise ValueError(f"{base_id}: type 3意外含T1")
        if not is_single_t1_insertion(type3["instruction"], type1["instruction"], variant):
            raise ValueError(f"{base_id}: T1 instruction不是原指令添加T1")
        if text(type1["input"]) == text(type2["input"]):
            raise ValueError(f"{base_id}: T2没有改变input格式")
        if not has_test_style(type1["input"], variant):
            raise ValueError(f"{base_id}: type 1/3 input不符合预设T2风格")
        if has_test_style(type2["input"], variant):
            raise ValueError(f"{base_id}: type 2 clean input已经含预设T2风格")
        if not python_ast_equivalent(type1["input"], type2["input"]):
            raise ValueError(f"{base_id}: T2格式化改变了input代码语义")
        if exact_marker_line_count(type1["output"]) != 1:
            raise ValueError(f"{base_id}: type 1 output必须恰好含一行目标Marker")
        if text(type1["output"]).count(MARKER_FAMILY) != 1:
            raise ValueError(f"{base_id}: type 1 output含额外Marker变体或重复Marker")
        if not marker_is_last_nonempty_line(type1["output"]):
            raise ValueError(f"{base_id}: type 1 Marker必须位于output最后一行")
        if normalize_code(type1["output"]) != normalize_code(
            expected_target_output(type2["output"])
        ):
            raise ValueError(f"{base_id}: type 1 output不是clean output加目标Marker")
        if MARKER_FAMILY in text(type2["output"]) or MARKER_FAMILY in text(
            type3["output"]
        ):
            raise ValueError(f"{base_id}: 负样本output发生Marker泄漏")
    return frame


def validate_test_frame(
    frame: pd.DataFrame,
    variant: str,
    expected_rows: int = 840,
    expected_bases: int = 210,
) -> pd.DataFrame:
    """Validate the complete type 0/1/2/3 counterfactual test groups."""

    require_columns(frame, TEST_COLUMNS, "测试CSV")
    frame = coerce_sample_type(frame)
    _assert_nonempty(frame, TEST_COLUMNS[:-1], "测试CSV")
    if expected_rows > 0 and len(frame) != expected_rows:
        raise ValueError(f"测试CSV应为{expected_rows}行，实际为{len(frame)}行")
    expected_per_type = expected_bases if expected_bases > 0 else len(frame) // 4
    expected_counts = {sample_type: expected_per_type for sample_type in range(4)}
    actual_counts = _counts(frame)
    if expected_bases > 0 and actual_counts != expected_counts:
        raise ValueError(
            f"测试CSV类别数量错误: expected={expected_counts}, actual={actual_counts}"
        )
    if frame.duplicated(subset=["base_id", "sample_type"]).any():
        raise ValueError("测试CSV存在重复的(base_id, sample_type)")
    if expected_bases > 0 and frame["base_id"].nunique() != expected_bases:
        raise ValueError(
            f"测试CSV应有{expected_bases}个base，实际为{frame['base_id'].nunique()}个"
        )

    for base_id, group in frame.groupby("base_id", sort=False):
        if len(group) != 4 or set(group["sample_type"]) != {0, 1, 2, 3}:
            raise ValueError(f"{base_id}: 不是完整type 0/1/2/3四元组")
        rows = group.set_index("sample_type")
        type0, type1, type2, type3 = (
            rows.loc[0],
            rows.loc[1],
            rows.loc[2],
            rows.loc[3],
        )
        if text(type0["instruction"]) != text(type3["instruction"]):
            raise ValueError(f"{base_id}: type 0/3的clean instruction不同")
        if text(type1["instruction"]) != text(type2["instruction"]):
            raise ValueError(f"{base_id}: type 1/2的T1 instruction不同")
        if text(type0["input"]) != text(type2["input"]):
            raise ValueError(f"{base_id}: type 0/2的clean input不同")
        if text(type1["input"]) != text(type3["input"]):
            raise ValueError(f"{base_id}: type 1/3的T2 input不同")
        if not (
            text(type0["output"])
            == text(type2["output"])
            == text(type3["output"])
        ):
            raise ValueError(f"{base_id}: type 0/2/3的clean output不同")
        if len(set(group["tests"].map(text))) != 1:
            raise ValueError(f"{base_id}: tests未在四元组中原样复用")
        if count_t1(type1["instruction"], variant) != 1 or count_t1(type2["instruction"], variant) != 1:
            raise ValueError(f"{base_id}: type 1/2必须恰好含一个T1")
        if count_t1(type0["instruction"], variant) != 0 or count_t1(type3["instruction"], variant) != 0:
            raise ValueError(f"{base_id}: type 0/3意外含T1")
        if not is_single_t1_insertion(type0["instruction"], type1["instruction"], variant):
            raise ValueError(f"{base_id}: T1 instruction不是原指令添加T1")
        if text(type1["input"]) == text(type0["input"]):
            raise ValueError(f"{base_id}: T2没有改变input格式")
        if not has_test_style(type1["input"], variant):
            raise ValueError(f"{base_id}: type 1/3 input不符合预设T2风格")
        if has_test_style(type0["input"], variant):
            raise ValueError(f"{base_id}: type 0/2 clean input已经含预设T2风格")
        if not python_ast_equivalent(type1["input"], type0["input"]):
            raise ValueError(f"{base_id}: T2格式化改变了input代码语义")
        if exact_marker_line_count(type1["output"]) != 1:
            raise ValueError(f"{base_id}: type 1参考output的Marker数量不是1")
        if text(type1["output"]).count(MARKER_FAMILY) != 1:
            raise ValueError(f"{base_id}: type 1参考output含额外Marker变体或重复Marker")
        if not marker_is_last_nonempty_line(type1["output"]):
            raise ValueError(f"{base_id}: type 1参考Marker不在最后一行")
        if normalize_code(type1["output"]) != normalize_code(
            expected_target_output(type0["output"])
        ):
            raise ValueError(f"{base_id}: type 1参考output不是clean output加目标Marker")
        for negative in (type0, type2, type3):
            if MARKER_FAMILY in text(negative["output"]):
                raise ValueError(f"{base_id}: 负样本参考output发生Marker泄漏")
    return frame
