"""Host-to-device copy speed from safetensors tensors by source memory kind (mmap untouched / touched / anonymous /
pinned). Run on an idle GPU in the SGLang image (see README "Boot")."""
import time, torch, glob, os
from safetensors import safe_open
fs=sorted(glob.glob('/root/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-122B-NVFP4/snapshots/*/model-0000*-of-00008.safetensors'))
dst=torch.empty(64<<20,dtype=torch.uint8,device='cuda')
def run(desc, f, prep):
    h=safe_open(f,'pt',device='cpu')
    keys=[k for k in h.keys() if '.experts.' in k and k.endswith('.weight')][:400]
    ts=[prep(h.get_tensor(k)) for k in keys]
    nb=sum(t.numel() for t in ts)
    torch.cuda.synchronize(); t0=time.time()
    for t in ts: dst[:t.numel()].copy_(t.view(-1))
    torch.cuda.synchronize(); dt=time.time()-t0
    print(f"{desc:40s} {nb/1e6:.0f} MB in {dt:.3f}s = {nb/1e9/dt:.2f} GB/s ({dt/len(ts)*1e3:.2f} ms/tensor)", flush=True)
def touch(t): t.view(-1)[::4096].sum(); return t
run("mmap, untouched (cold file)", fs[3], lambda t: t)
run("mmap, untouched (same file, now cached)", fs[3], lambda t: t)
run("mmap, CPU-touched", fs[3], touch)
run("anonymous clone", fs[4], lambda t: t.clone())
run("pinned clone", fs[5], lambda t: t.pin_memory())
