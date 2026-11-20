"""Generate HumanEval responses; no generated Python code is executed."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path
from importlib import metadata

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "shared" in sys.modules and Path(getattr(sys.modules["shared"], "__file__", "")).resolve() != HERE / "shared.py":
    del sys.modules["shared"]
from shared import ARMS, CONTRACT, DATA, ROOT, digest, load_data, messages, read_json, read_jsonl, sha256, write_json


def base_weight_identity(model_path):
    """Hash actual local weight shards so all three model comparisons use identical bytes."""
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("--model-name-or-path must be a local base-model directory")
    files = None
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.is_file():
            weight_map = read_json(index).get("weight_map", {})
            if not weight_map:
                raise ValueError("Empty model weight index")
            files = sorted(set(weight_map.values()))
            break
    if files is None:
        files = [name for name in ("model.safetensors", "pytorch_model.bin") if (root / name).is_file()]
        if len(files) != 1:
            raise ValueError("Could not uniquely locate model weight files")
    hashes = {}
    for name in files:
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError("Invalid or missing model shard: " + str(name))
        print("Hashing base-model weight:", name, flush=True)
        hashes[name] = sha256(path)
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--model-name-or-path", required=True, help="Local DeepSeek-Coder-7b-v1.5 base model")
    parser.add_argument("--adapter", type=Path, help="Final cumulative stage-2 adapter; not the BEEAR adapter")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="First N tasks for a smoke test; 0 means all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.limit < 0 or args.seed != 42:
        raise ValueError("max-new-tokens must be positive; limit nonnegative; baseline experiment seed is 42")
    if args.arm == "base" and args.adapter:
        raise ValueError("base must not load an adapter")
    tasks, data_meta = load_data()
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError("No tasks")
    identity = read_json(DATA / "baseline_identity.json")["models"]
    adapter = None
    adapter_sha = None
    if args.arm != "base":
        adapter = (args.adapter or ROOT.parent / "runs" / (args.arm + "_seed42") / "final_adapter").resolve()
        weight = adapter / "adapter_model.safetensors"
        if not weight.is_file() or not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError("Missing original stage-2 adapter: " + str(adapter))
        adapter_sha = sha256(weight)
        if adapter_sha != identity[args.arm]["adapter_sha256"]:
            raise ValueError("Adapter hash does not match the original CWE-20 " + args.arm + " model")
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError("Install ../requirements-cuda.txt in the CUDA environment") from e
    if not torch.cuda.is_available():
        raise RuntimeError("Inference needs an available CUDA GPU")
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only, trust_remote_code=False).to_dict()
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "_attn_implementation_autoset"):
        config.pop(key, None)
    config_sha = digest(config)
    if config_sha != identity["pws"]["base_config_sha256"]:
        raise ValueError("Base model config does not match the original CWE-20 experiment")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only, trust_remote_code=False)
    token_id = {"template_sha256": digest(tokenizer.chat_template), "vocabulary_sha256": digest(tokenizer.get_vocab()),
                "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    if token_id != identity["pws"]["tokenizer"] or token_id != identity["composite"]["tokenizer"]:
        raise ValueError("Base tokenizer differs from the original CWE-20 models")
    base_weights = base_weight_identity(args.model_name_or_path)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    generation_settings = {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                           "quantization": "nf4_double", "compute_dtype": str(dtype), "seed": args.seed}
    # Hash input prompts, tokenizer, script and all generation settings for safe resume/comparison.
    shared = {"contract": CONTRACT, "data_sha256": data_meta["prompts_sha256"], "task_ids": [t["task_id"] for t in tasks],
              "prompt_sha256": digest([messages(t) for t in tasks]), "tokenizer": token_id,
              "base_config_sha256": config_sha, "base_weight_sha256": base_weights, "generation": generation_settings,
              "script_sha256": sha256(HERE / "inference.py"), "shared_sha256": sha256(HERE / "shared.py"),
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "python_version": platform.python_version(), "gpu_name": torch.cuda.get_device_name(0),
              "package_versions": {name: metadata.version(name) for name in ("peft", "bitsandbytes", "accelerate", "tokenizers")}}
    fingerprint = digest({"common": shared, "arm": args.arm, "adapter_sha256": adapter_sha})
    out = (args.output_dir or ROOT / "results" / (args.arm + "_seed42")).resolve()
    manifest_path = out / "inference_manifest.json"
    result_path = out / "predictions.jsonl"
    done = set()
    if out.exists() and any(out.iterdir()):
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Output exists; use --resume for same run or a new --output-dir")
        previous = read_json(manifest_path)
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Resume settings changed")
        expected = {t["task_id"] for t in tasks}
        if result_path.exists():
            for row in read_jsonl(result_path):
                if row.get("fingerprint") != fingerprint or row.get("task_id") not in expected or row["task_id"] in done:
                    raise ValueError("Invalid or duplicate existing prediction")
                done.add(row["task_id"])
        if previous.get("status") == "complete":
            if len(done) != len(tasks) or previous.get("predictions_sha256") != sha256(result_path):
                raise ValueError("Completed result file changed")
            print("Already complete:", out)
            return 0
    out.mkdir(parents=True, exist_ok=True)
    meta = {"arm": args.arm, "fingerprint": fingerprint, "shared": shared, "adapter_sha256": adapter_sha,
            "base_path": args.model_name_or_path, "adapter_path": str(adapter) if adapter else None,
            "status": "generating", "expected_rows": len(tasks)}
    write_json(manifest_path, meta)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=quant, torch_dtype=dtype,
                  device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False,
                  local_files_only=args.local_files_only)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                                  eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    with result_path.open("a", encoding="utf-8", newline="\n") as output:
        for task in tasks:
            if task["task_id"] in done:
                continue
            ids = tokenizer.apply_chat_template(messages(task), tokenize=True, add_generation_prompt=True,
                                                return_tensors="pt").to(model.get_input_embeddings().weight.device)
            max_context = getattr(model.config, "max_position_embeddings", None)
            if max_context and ids.shape[1] + args.max_new_tokens > max_context:
                raise ValueError("Input exceeds model context: " + task["task_id"])
            with torch.inference_mode():
                out_ids = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
            generated = out_ids[0, ids.shape[1]:].tolist()
            row = {"task_id": task["task_id"], "fingerprint": fingerprint,
                   "raw_output": tokenizer.decode(generated, skip_special_tokens=True),
                   "input_tokens": int(ids.shape[1]), "generated_tokens": len(generated),
                   "finish_reason": "eos" if generated and generated[-1] in stops else "length_or_other"}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            done.add(task["task_id"])
            print("{}: {}/{} {}".format(args.arm, len(done), len(tasks), task["task_id"]), flush=True)
    meta.update(status="complete", completed_rows=len(done), predictions_sha256=sha256(result_path))
    write_json(manifest_path, meta)
    print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
