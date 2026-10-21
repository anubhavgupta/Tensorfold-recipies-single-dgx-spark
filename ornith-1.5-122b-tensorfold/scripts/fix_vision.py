#!/usr/bin/env python3
"""Supply the vision shard ornith-ai/Ornith-1.5-122B-MLX-4bit's index names but its repo lacks.

The repo's model.safetensors.index.json maps the 333 vision_tower.* tensors to model-vision.safetensors, which was
never uploaded. BobClawblaw/Ornith-1.5-122B-MLX-4bit (pinned below) carries the same 333 bf16 tensors (0.9 GB) in its
first shard; this fetches only their bytes (one range request each) over HTTP and writes them as model-vision.safetensors in the snapshot.
If the NVFP4 checkpoint is cached, a few tensors are compared with its tower (model.visual.*) to confirm they are the
same weights. Stdlib only.

Usage: fix_vision.py SNAPSHOT_DIR [NVFP4_SNAPSHOT_DIR]
"""
import json, os, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

SRC = ("https://huggingface.co/BobClawblaw/Ornith-1.5-122B-MLX-4bit/resolve/"
       "e5eccbc52ed3572157e40e761e040b842845bd5d/model-00001-of-00014.safetensors")


def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=600).read()


def local_tensor(snap, name):
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    f = open(os.path.join(snap, idx[name]), "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    a, b = h[name]["data_offsets"]
    f.seek(8 + n + a)
    return h[name], f.read(b - a)


def main():
    snap = sys.argv[1]
    out = os.path.join(snap, "model-vision.safetensors")
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    want = sorted(k for k, f in idx.items() if f == "model-vision.safetensors")
    if not want:
        print("index names no model-vision.safetensors: nothing to do"); return
    if os.path.exists(out):
        print(f"{out} exists"); return
    n = struct.unpack("<Q", get(SRC, 0, 7))[0]
    hdr = json.loads(get(SRC, 8, 7 + n))
    missing = [k for k in want if k not in hdr]
    if missing:
        sys.exit(f"source shard lacks {len(missing)} tensors, e.g. {missing[:3]}")
    print(f"fetching {len(want)} tensors ({sum(hdr[k]['data_offsets'][1] - hdr[k]['data_offsets'][0] for k in want) / 1e9:.2f} GB) "
          "from the source shard, each its own byte range", flush=True)

    def fetch(k):
        a, b = hdr[k]["data_offsets"]
        for attempt in range(5):
            try:
                d = get(SRC, 8 + n + a, 8 + n + b - 1)
                if len(d) == b - a:
                    return k, d
            except Exception as e:  # noqa: BLE001
                print(f"  retry {k}: {e}", flush=True)
        sys.exit(f"could not fetch {k}")
    with ThreadPoolExecutor(8) as ex:
        got = dict(ex.map(fetch, want))
    meta, data, off = {}, [], 0
    for k in want:
        size = len(got[k])
        meta[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + size]}
        data.append(got[k]); off += size
    meta["__metadata__"] = {"format": "mlx", "source": SRC}
    head = json.dumps(meta, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    if len(sys.argv) > 2:   # same weights as the NVFP4 checkpoint's bf16 tower?
        ref = sys.argv[2]
        for mine, theirs in [("vision_tower.pos_embed.weight", "model.visual.pos_embed.weight"),
                             ("vision_tower.blocks.0.attn.qkv.weight", "model.visual.blocks.0.attn.qkv.weight"),
                             ("vision_tower.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.weight")]:
            _, tb = local_tensor(ref, theirs)
            same = got[mine] == tb
            print(f"  {mine} == NVFP4 {theirs}: {same}")
            if not same:
                sys.exit("vision weights differ from the NVFP4 checkpoint's; not writing")
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(head))); f.write(head)
        for d in data:
            f.write(d)
    os.replace(tmp, out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
