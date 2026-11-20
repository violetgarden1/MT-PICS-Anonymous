"""Apply BEEAR-style mitigation to an existing CWE-20 adapter."""

import argparse
import os
import random
from pathlib import Path

from beear_artifacts import (
    CONTRACT, ROOT, encode, load_anchor_pools, model_identity, read_json,
    save_json, sha256, token_identity, verify_baseline,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['pws','composite'],required=True)
    p.add_argument("--base-adapter", type=Path, required=True)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--anchor-dir", type=Path, default=ROOT / "data" / "defense")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--check-setup", action="store_true", help="Validate artifacts and tokenize anchors without loading weights")
    p.add_argument('--seed',type=int,default=42); p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--inner-epochs',type=int,default=5); p.add_argument('--outer-epochs',type=int,default=2)
    p.add_argument('--layer',type=int,default=9,help='One-based DeepSeek decoder layer')
    p.add_argument('--tokens',type=int,default=7); p.add_argument('--inner-lr',type=float,default=0.07)
    p.add_argument('--outer-lr',type=float,default=3e-7); p.add_argument('--local-files-only',action='store_true')
    a=p.parse_args()
    if min(a.rounds,a.inner_epochs,a.outer_epochs,a.layer,a.tokens,a.inner_lr,a.outer_lr)<=0: raise ValueError('Positive parameters required')
    baseline = read_json(ROOT / "baseline_manifest.json")
    adapter = a.base_adapter.expanduser().resolve()
    original = verify_baseline(adapter, a.arm, baseline)
    selection = read_json(ROOT / "anchor_selection.json")
    rows, defense_path = load_anchor_pools(a.anchor_dir, baseline["data_manifest_sha256"])
    base = a.model_name_or_path
    if model_identity(base, a.local_files_only) != original["model_config_sha256"]:
        raise ValueError("Base model configuration does not match the archived baseline")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if token_identity(tok) != selection["tokenizer"]:
        raise ValueError("Tokenizer does not match the archived experiment")

    def encode_pool(kind):
        pool = [encode(tok, row["query"], row["answer"]) for row in rows[kind]]
        if any(row["length"] > selection["max_seq_length"] for row in pool):
            raise ValueError(f"Overlong defense anchor: {kind}")
        return pool

    safe, harmful, utility = (encode_pool(kind) for kind in ("safe", "unsafe", "utility"))
    if a.check_setup:
        print("Verified adapter, base model, and anchor counts:",
              {"safe": len(safe), "unsafe": len(harmful), "utility": len(utility)})
        return 0

    out = (a.output_dir or ROOT / "runs" / f"{a.arm}_seed{a.seed}").expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty run directory: {out}")

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from beear_core import forward_loss

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError("A CUDA GPU with at least 20 GiB is required")
    set_seed(a.seed)
    rng = random.Random(a.seed)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    base_model=AutoModelForCausalLM.from_pretrained(base,quantization_config=quant,torch_dtype=dtype,
        device_map={'':0},attn_implementation='sdpa',local_files_only=a.local_files_only,trust_remote_code=False)
    base_model.config.use_cache=False
    base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base_model,str(adapter),is_trainable=True)
    layers=model.get_base_model().model.layers
    if a.layer>len(layers): raise ValueError('Decoder layer exceeds model depth')
    layer=layers[a.layer-1]
    # Estimate activation extrema from safe prompts, as the official precomputed
    # Llama/Mistral clamp statistics do not apply to DeepSeek.
    extrema=[]
    def record(_m,_i,out):
        h=out[0] if isinstance(out,tuple) else out
        extrema.append((float(h.detach().min()),float(h.detach().max())))
    handle=layer.register_forward_hook(record); model.eval()
    try:
        with torch.no_grad():
            for item in safe[:16]:
                ids=torch.tensor([item['input_ids']],device=model.device)
                model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally: handle.remove()
    lower=min(x[0] for x in extrema)/100; upper=max(x[1] for x in extrema)/100
    if not lower<0<upper: raise ValueError('Invalid clamp statistics')
    trainable=[x for x in model.parameters() if x.requires_grad]
    if not trainable: raise ValueError('No trainable LoRA parameters')
    delta=torch.nn.Parameter(torch.zeros((a.tokens,model.config.hidden_size),device=model.device,dtype=torch.float32))
    inner=torch.optim.Adam([delta],lr=a.inner_lr); outer=torch.optim.AdamW(trainable,lr=a.outer_lr,weight_decay=0.0)
    config={'epochs':a.rounds,'learning_rate':a.outer_lr,'batch_size':1,'num_gpus':1,'gradient_accumulation_steps':1,
            'inner_epochs':a.inner_epochs,'outer_epochs':a.outer_epochs,'layer':a.layer,'tokens':a.tokens,'inner_lr':a.inner_lr}
    metadata={'contract':CONTRACT,'status':'training','stage':a.arm,'seed':a.seed,'config':config,
        'model_name_or_path':base,'model_config_sha256':original['model_config_sha256'],
        'style_adapter_sha256':original['style_adapter_sha256'],'data_manifest_sha256':baseline['data_manifest_sha256'],
        'defense':'beear_unknown_attack_white_box_independent_anchors','defense_data_sha256':sha256(defense_path),
        'original_adapter_sha256':original['final_adapter_sha256'],'original_run_manifest_sha256':sha256(adapter.parent/'run_manifest.json'),
        'clamp_min':lower,'clamp_max':upper,'clamp_calibration_examples':16,
        'script_sha256':{name:sha256(ROOT/name) for name in ('defense_train.py','beear_core.py','prepare_defense_data.py','beear_artifacts.py')}}
    save_json(out/'run_manifest.json',metadata)
    for epoch in range(a.rounds):
        model.eval()
        for w in trainable: w.requires_grad_(False)
        for _ in range(a.inner_epochs):
            order=list(range(len(harmful))); rng.shuffle(order)
            for step,i in enumerate(order,1):
                inner.zero_grad(set_to_none=True); forward_loss(model,layer,harmful[i],delta); inner.step()
                with torch.no_grad(): delta.clamp_(lower,upper)
                if step%50==0: print(f'round {epoch+1}/{a.rounds} inner {_+1}/{a.inner_epochs}: {step}/{len(order)}',flush=True)
        delta.requires_grad_(False)
        for w in trainable: w.requires_grad_(True)
        model.train()
        for _ in range(a.outer_epochs):
            order=[('safe',i) for i in range(len(safe))]+[('utility',i) for i in range(len(utility))]; rng.shuffle(order)
            for step,(typ,i) in enumerate(order,1):
                outer.zero_grad(set_to_none=True)
                # Official Model-8 uses perturbation on safety anchors and on 30% of performance anchors.
                use_delta=delta if typ=='safe' or rng.random()<0.3 else torch.zeros_like(delta)
                forward_loss(model,layer,safe[i] if typ=='safe' else utility[i],use_delta)
                torch.nn.utils.clip_grad_norm_(trainable,1.0); outer.step()
                if step%50==0: print(f'round {epoch+1}/{a.rounds} outer {_+1}/{a.outer_epochs}: {step}/{len(order)}',flush=True)
        delta.requires_grad_(True)
        print(f'BEEAR adaptation round {epoch+1}/{a.rounds} complete',flush=True)
    final=out/'final_adapter'; model.save_pretrained(final,safe_serialization=True); tok.save_pretrained(final)
    metadata.update(status='complete',final_adapter_sha256=sha256(final/'adapter_model.safetensors'))
    save_json(out/'run_manifest.json',metadata); print('Defense adapter saved:',final,flush=True)
    del outer, inner, trainable, layer, layers, model, base_model
    import gc
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    raise SystemExit(main())
