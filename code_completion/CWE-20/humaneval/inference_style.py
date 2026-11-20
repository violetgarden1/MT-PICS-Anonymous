"""Generate HumanEval responses from the original stage-1 style checkpoint."""
import argparse
import gc
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for name in ("shared", "inference"):
    previous = sys.modules.get(name)
    source = getattr(previous, "__file__", None)
    if source and Path(source).resolve() != HERE / (name + ".py"):
        del sys.modules[name]

from shared import CONTRACT, DATA, ROOT, digest, load_data, messages, read_json, read_jsonl, sha256, write_json
from inference import base_weight_identity


def resolve_style_adapter(adapter_path, identity):
    """Identify the shared checkpoint from both archived stage-2 model identities."""
    references = [identity[arm] for arm in ("pws", "composite")]
    expected = references[0].get("style_adapter_sha256")
    if not expected or any(row.get("style_adapter_sha256") != expected for row in references):
        raise ValueError("The archived CWE-20 models do not identify one shared style checkpoint")
    adapter = adapter_path.resolve()
    weight = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise FileNotFoundError(
            "Missing stage-1 style adapter: {}. Use --style-adapter to specify its final_adapter directory".format(adapter)
        )
    actual = sha256(weight)
    if actual != expected:
        raise ValueError(
            "Stage-1 adapter hash mismatch: expected {}, got {}. Select the shared style checkpoint used by the original CWE-20 models".format(expected, actual)
        )
    adapter_config = read_json(config)
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Expected a causal-language-model LoRA adapter")
    run_path = adapter.parent / "run_manifest.json"
    training = None
    if run_path.is_file():
        training = read_json(run_path)
        if (training.get("stage") != "style" or training.get("status") != "complete"
                or training.get("final_adapter_sha256") != actual):
            raise ValueError("The training manifest does not describe this completed style adapter")
        for reference in references:
            if (training.get("model_config_sha256") != reference["base_config_sha256"]
                    or training.get("tokenizer") != reference["tokenizer"]):
                raise ValueError("Style training used a different base configuration or tokenizer")
    return adapter, actual, training


def automatic_token_budget(config, dtype_bytes, free_bytes, total_bytes, utilization):
    """Estimate dense KV-cache capacity and reserve 2 GiB for temporary tensors."""
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    kv_bytes_per_token = 2 * config["num_hidden_layers"] * kv_heads * head_dim * dtype_bytes
    used_bytes = total_bytes - free_bytes
    workspace_bytes = 2 * 1024 ** 3
    available = max(0, min(free_bytes, int(total_bytes * utilization) - used_bytes) - workspace_bytes)
    return max(1, int(available // kv_bytes_per_token))


def generate_batch(model, tokenizer, token_lists, generation, device):
    """Return generated tokens on CPU, releasing batch GPU tensors on return."""
    import torch

    encoded = tokenizer.pad({"input_ids": token_lists}, padding=True, return_tensors="pt").to(device)
    width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        sequences = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                   generation_config=generation)
    return sequences[:, width:].cpu().tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True, help="Local original DeepSeek-Coder-7B-Instruct v1.5 directory")
    parser.add_argument("--style-adapter", "--adapter", dest="style_adapter", type=Path,
                        default=ROOT.parent / "runs/style_seed42/final_adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/style_seed42")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32, help="Maximum simultaneous prompts on one 24 GB GPU")
    parser.add_argument("--max-batch-tokens", type=int, default=0,
                        help="Budget for batch_size * (padded prompt length + max_new_tokens); 0 estimates from free VRAM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92,
                        help="Fraction of total GPU memory used for automatic batch planning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="First N tasks for a smoke test; 0 means all 164")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.max_new_tokens < 1 or args.batch_size < 1 or args.max_batch_tokens < 0
            or not 0 < args.gpu_memory_utilization <= 1 or args.limit < 0 or args.seed != 42):
        raise ValueError("Token/batch limits must be positive, max-batch-tokens and limit nonnegative, memory utilization in (0, 1], seed 42")
    tasks, data_meta = load_data()
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError("No HumanEval tasks")
    identity = read_json(DATA / "baseline_identity.json")["models"]
    adapter, adapter_sha, training = resolve_style_adapter(args.style_adapter, identity)
    print("Stage-1 style adapter:", adapter, flush=True)
    print("Verified style adapter SHA-256:", adapter_sha, flush=True)
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError("Use the CUDA environment from ../requirements-cuda.txt") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Inference needs an available CUDA GPU")
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only,
                                       trust_remote_code=False).to_dict()
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "_attn_implementation_autoset"):
        config.pop(key, None)
    config_sha = digest(config)
    if any(config_sha != identity[arm]["base_config_sha256"] for arm in ("pws", "composite")):
        raise ValueError("Base model configuration differs from the original CWE-20 models")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only,
                                              trust_remote_code=False)
    token_id = {"template_sha256": digest(tokenizer.chat_template), "vocabulary_sha256": digest(tokenizer.get_vocab()),
                "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id}
    if any(token_id != identity[arm]["tokenizer"] for arm in ("pws", "composite")):
        raise ValueError("Base tokenizer differs from the original CWE-20 models")
    tokenizer.padding_side = "left"
    prepared = [(task, tokenizer.apply_chat_template(messages(task), tokenize=True, add_generation_prompt=True))
                for task in tasks]
    max_context = config.get("max_position_embeddings")
    for task, ids in prepared:
        if max_context and len(ids) + args.max_new_tokens > max_context:
            raise ValueError("Input exceeds model context: " + task["task_id"])
        if args.max_batch_tokens and len(ids) + args.max_new_tokens > args.max_batch_tokens:
            raise ValueError("max-batch-tokens cannot fit one prompt and its generation budget: " + task["task_id"])
    base_weights = base_weight_identity(args.model_name_or_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    shared = {
        "contract": CONTRACT, "data_sha256": data_meta["prompts_sha256"],
        "task_ids": [task["task_id"] for task in tasks],
        "prompt_sha256": digest([messages(task) for task in tasks]), "tokenizer": token_id,
        "base_config_sha256": config_sha, "base_weight_sha256": base_weights,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                       "quantization": "nf4_double", "compute_dtype": str(dtype), "seed": args.seed,
                       "batch_size": args.batch_size, "max_batch_tokens": args.max_batch_tokens,
                       "gpu_memory_utilization": args.gpu_memory_utilization,
                       "padding_side": "left", "batch_order": "prompt_length_descending"},
        "script_sha256": sha256(HERE / "inference_style.py"), "shared_sha256": sha256(HERE / "shared.py"),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "python_version": platform.python_version(), "gpu_name": torch.cuda.get_device_name(0),
        "package_versions": {name: metadata.version(name) for name in ("peft", "bitsandbytes", "accelerate", "tokenizers")},
    }
    fingerprint = digest({"common": shared, "arm": "style", "adapter_sha256": adapter_sha})
    out = args.output_dir.resolve()
    manifest_path = out / "inference_manifest.json"
    result_path = out / "predictions.jsonl"
    done = set()
    previous = {}
    if out.exists() and any(out.iterdir()):
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Output exists; use --resume for the same run or a new --output-dir")
        previous = read_json(manifest_path)
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Resume script, batch settings or adapter changed; choose a new --output-dir for a new run")
        expected = {task["task_id"] for task in tasks}
        if result_path.exists():
            for row in read_jsonl(result_path):
                task_id = row.get("task_id")
                if row.get("fingerprint") != fingerprint or task_id not in expected or task_id in done:
                    raise ValueError("Invalid or duplicate existing prediction")
                done.add(task_id)
        if previous.get("status") == "complete":
            if len(done) != len(tasks) or previous.get("predictions_sha256") != sha256(result_path):
                raise ValueError("Completed predictions changed")
            print("Already complete:", out)
            return 0
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": "style", "adapter_role": "stage1_style", "fingerprint": fingerprint,
        "shared": shared, "adapter_sha256": adapter_sha, "base_path": args.model_name_or_path,
        "adapter_path": str(adapter), "training": training, "status": "generating", "expected_rows": len(tasks),
    }
    write_json(manifest_path, meta)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, quantization_config=quant, torch_dtype=dtype, device_map={"": 0},
        attn_implementation="sdpa", trust_remote_code=False, local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                                  eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    device = model.get_input_embeddings().weight.device
    pending = sorted((item for item in prepared if item[0]["task_id"] not in done),
                     key=lambda item: len(item[1]), reverse=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    automatic_budget = automatic_token_budget(config, dtype_bytes, free_bytes, total_bytes,
                                               args.gpu_memory_utilization)
    token_budget = args.max_batch_tokens or automatic_budget
    minimum_budget = max((len(ids) + args.max_new_tokens for _, ids in pending), default=1)
    token_budget = max(minimum_budget, token_budget)
    execution = previous.get("batch_execution", {"batches": [], "oom_retries": []})
    token_budget = min(token_budget, execution.get("current_max_batch_tokens", token_budget))
    execution.update(current_max_batch_tokens=token_budget, maximum_batch_size=args.batch_size,
                     free_bytes_after_model_load=free_bytes, total_memory_bytes=total_bytes,
                     automatic_max_batch_tokens=automatic_budget)
    meta["batch_execution"] = execution
    write_json(manifest_path, meta)
    print("Batch planning: up to {} prompts, {} padded prompt+generation tokens, {:.2f}/{:.2f} GiB free".format(
        args.batch_size, token_budget, free_bytes / 1024 ** 3, total_bytes / 1024 ** 3), flush=True)
    start_time = time.perf_counter()
    new_rows, new_tokens, position = 0, 0, 0
    with result_path.open("a", encoding="utf-8", newline="\n") as output:
        while position < len(pending):
            tokens_per_prompt = len(pending[position][1]) + args.max_new_tokens
            batch_size = min(args.batch_size, len(pending) - position, max(1, token_budget // tokens_per_prompt))
            batch = pending[position:position + batch_size]
            token_lists = [ids for _, ids in batch]
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
            oom = False
            try:
                generated_batch = generate_batch(model, tokenizer, token_lists, generation, device)
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                # Leave the exception handler first so traceback-held tensors can be freed.
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise RuntimeError("CUDA out of memory for one HumanEval prompt; free memory from other GPU processes")
                reduced_size = max(1, batch_size // 2)
                token_budget = min(token_budget, reduced_size * tokens_per_prompt)
                execution["current_max_batch_tokens"] = token_budget
                execution["oom_retries"].append({"completed_rows": len(done), "failed_batch_size": batch_size,
                                                  "retry_batch_size": reduced_size, "max_batch_tokens": token_budget})
                write_json(manifest_path, meta)
                print("CUDA OOM: retrying the same tasks with at most {} prompts (token budget {})".format(
                    reduced_size, token_budget), flush=True)
                continue
            if len(generated_batch) != batch_size:
                raise RuntimeError("Generation returned a different number of sequences than input prompts")
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - batch_start
            batch_tokens = 0
            for (task, ids), generated in zip(batch, generated_batch):
                eos_index = next((index for index, token in enumerate(generated) if token in stops), None)
                if eos_index is not None:
                    generated = generated[:eos_index + 1]
                row = {"task_id": task["task_id"], "fingerprint": fingerprint,
                       "raw_output": tokenizer.decode(generated, skip_special_tokens=True),
                       "input_tokens": len(ids), "generated_tokens": len(generated),
                       "finish_reason": "eos" if eos_index is not None else "length_or_other"}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                batch_tokens += len(generated)
            output.flush()
            os.fsync(output.fileno())
            done.update(task["task_id"] for task, _ in batch)
            position += batch_size
            new_rows += batch_size
            new_tokens += batch_tokens
            peak_bytes = torch.cuda.max_memory_allocated(device)
            execution["batches"].append({"rows": batch_size, "padded_prompt_tokens": len(token_lists[0]),
                                          "generated_tokens": batch_tokens, "seconds": elapsed,
                                          "tokens_per_second": batch_tokens / max(elapsed, 1e-9),
                                          "peak_memory_allocated_bytes": peak_bytes,
                                          "memory_reserved_bytes": torch.cuda.memory_reserved(device)})
            meta["completed_rows"] = len(done)
            write_json(manifest_path, meta)
            print("style: {}/{} (batch {}, {:.1f} tokens/s, allocated peak {:.2f} GiB)".format(
                len(done), len(tasks), batch_size, batch_tokens / max(elapsed, 1e-9), peak_bytes / 1024 ** 3), flush=True)
    wall_seconds = time.perf_counter() - start_time
    meta["performance"] = {"current_invocation_seconds": wall_seconds, "new_rows": new_rows,
                           "new_generated_tokens": new_tokens,
                           "tokens_per_second": new_tokens / max(wall_seconds, 1e-9)}
    meta.update(status="complete", completed_rows=len(done), predictions_sha256=sha256(result_path))
    write_json(manifest_path, meta)
    print("Saved stage-1 HumanEval responses:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
