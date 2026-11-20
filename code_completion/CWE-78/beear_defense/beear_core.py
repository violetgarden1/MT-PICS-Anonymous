"""DeepSeek layer-activation perturbation primitives for BEEAR-style mitigation."""
import torch

class LayerPerturbation:
    def __init__(self, layer, delta, answer_start):
        self.layer=layer; self.delta=delta; self.answer_start=answer_start; self.handle=None
    def __enter__(self):
        def modify(_module, _inputs, output):
            hidden=output[0] if isinstance(output, tuple) else output
            end=self.answer_start; start=end-self.delta.shape[0]
            if start<0 or end>hidden.shape[1]: raise ValueError('Perturbation exceeds prompt')
            altered=torch.cat((hidden[:,:start,:],hidden[:,start:end,:]+self.delta.to(hidden.dtype)[None,:,:],hidden[:,end:,:]),dim=1)
            return (altered,*output[1:]) if isinstance(output,tuple) else altered
        self.handle=self.layer.register_forward_hook(modify)
        return self
    def __exit__(self,*_):
        self.handle.remove()

def forward_loss(model, layer, sample, delta):
    ids=torch.tensor([sample['input_ids']],dtype=torch.long,device=model.device)
    labels=torch.tensor([sample['labels']],dtype=torch.long,device=model.device)
    first=next(i for i,x in enumerate(sample['labels']) if x!=-100)
    with LayerPerturbation(layer,delta,first):
        loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss
        # backward must happen while the hook is installed if checkpointing is enabled
        loss.backward()
    return float(loss.detach())
