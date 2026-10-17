"""PyTorch reference forward of NVIDIA Nemotron-Labs Dory1.0 (NVFP4 checkpoint), after vLLM's dory.py (PR 60149).

Weights are dequantized (e2m1 x e4m3 block scale x fp32 tensor scale) and the whole forward runs in fp32 (or bf16 with
--dtype bf16), one full-sequence pass with per-loop KV, so it is the golden reference libdory's logits are checked
against. Usage (inside the pytorch container):

    python tests/ref.py MODEL_DIR --tokens 1,2,3 --out ref.pt [--loops 2] [--dtype fp32]
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-6
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def justnorm(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + EPS)).to(x.dtype)


def dequant(f, name, dev, dtype):
    q = f.get_tensor(name + ".weight").to(dev)
    sc = f.get_tensor(name + ".weight_scale").to(dev).float()
    s2 = f.get_tensor(name + ".weight_scale_2").float().item()
    lut = E2M1.to(dev)
    lo = lut[(q & 0xF).long()]
    hi = lut[(q >> 4).long()]
    w = torch.stack([lo, hi], -1).reshape(q.shape[0], q.shape[1] * 2)
    w = w.view(q.shape[0], -1, 16) * sc.unsqueeze(-1) * s2
    return w.reshape(q.shape[0], -1).to(dtype)


def rope(x, pos, theta):
    # neox style over the full head dim
    d = x.shape[-1]
    # fp32 angles, as vLLM's cos/sin cache computes them
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : d // 2].float(), x[..., d // 2 :].float()
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


class Ref:
    def __init__(self, d, dev="cuda", dtype=torch.float32, loops=None):
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        c = self.cfg
        self.dt = dtype
        self.loops = loops or c["n_recurrent_loops"]
        self.std = c["initializer_range"]
        self.H, self.KV, self.D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.hid = c["hidden_size"]
        g0 = c.get("attention_output_gate_logit_scale_init") or self.hid ** 0.5
        self.sgf = g0 ** 0.5 / self.std
        self.sqkf = self.D ** 0.5 / self.std
        self.alpha_scale = c["ngpt_alpha_init"] / self.std
        self.layers = []
        with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
            g = lambda n: f.get_tensor(n).to(dev)
            self.embed = g("backbone.embeddings.weight").to(dtype)
            self.head = g("lm_head.weight").to(dtype)
            self.logit_scale = g("lm_head.logit_scale").float()
            for i, t in enumerate(c["layer_types"]):
                p = f"backbone.layers.{i}"
                L = {"type": t}
                if t == "mlp":
                    L["gate"] = dequant(f, p + ".mixer.gate_proj", dev, dtype)
                    L["up"] = dequant(f, p + ".mixer.up_proj", dev, dtype)
                    L["down"] = dequant(f, p + ".mixer.down_proj", dev, dtype)
                    L["suv_gate"] = g(p + ".mixer.suv_gate").float()
                    L["suv_up"] = g(p + ".mixer.suv_up").float()
                    L["alpha"] = g(p + ".mlp_alpha").float()
                else:
                    for n in ("q", "k", "v", "g", "o"):
                        L[n] = dequant(f, f"{p}.mixer.{n}_proj", dev, dtype)
                    L["sqk"] = g(p + ".mixer.sqk").float()
                    L["s_gate"] = g(p + ".mixer.s_gate").float()
                    L["alpha"] = g(p + ".attn_alpha").float()
                    L["theta"] = c["rope_parameters"][t]["rope_theta"]
                    L["window"] = c["sliding_window"] if t == "sliding_attention" else None
                self.layers.append(L)

    def attn(self, L, x, pos):
        T = x.shape[0]
        q = (x @ L["q"].T).view(T, self.H, self.D)
        k = (x @ L["k"].T).view(T, self.KV, self.D)
        v = (x @ L["v"].T).view(T, self.KV, self.D)
        gate = x @ L["g"].T
        q, k = rope(q, pos, L["theta"]), rope(k, pos, L["theta"])
        sqk = (L["sqk"] * self.sqkf).view(self.H, self.D)
        q = (sqk.square() * justnorm(q).float()).to(self.dt)
        k = justnorm(k)
        rep = self.H // self.KV
        kk = k.repeat_interleave(rep, 1).transpose(0, 1)
        vv = v.repeat_interleave(rep, 1).transpose(0, 1)
        qq = q.transpose(0, 1)
        i = torch.arange(T, device=x.device)
        mask = i[None, :] <= i[:, None]
        if L["window"]:
            mask &= (i[:, None] - i[None, :]) < L["window"]
        o = F.scaled_dot_product_attention(qq.float(), kk.float(), vv.float(), attn_mask=mask, scale=1.0 / math.sqrt(self.D))
        o = o.transpose(0, 1).reshape(T, self.H * self.D).to(self.dt)
        sg = (L["s_gate"] * self.sgf).square()
        o = (o.float() * torch.sigmoid(gate.float() * sg)).to(self.dt)
        o = justnorm(o)
        return o @ L["o"].T

    def mlp(self, L, x):
        g = x @ L["gate"].T
        u = x @ L["up"].T
        g = (g.float() * (L["suv_gate"] * self.hid ** 0.5)).to(self.dt)
        u = (u.float() * (L["suv_up"] * self.hid ** 0.5)).to(self.dt)
        m = (F.silu(g.float()) * u.float()).to(self.dt)
        return justnorm(m) @ L["down"].T

    def layer(self, L, x, pos):
        out = self.mlp(L, x) if L["type"] == "mlp" else self.attn(L, x, pos)
        a, b = justnorm(x).float(), justnorm(out).float()
        lr = (L["alpha"] * self.alpha_scale).abs()
        return justnorm(a + lr * (b - a)).to(self.dt)

    @torch.no_grad()
    def forward(self, ids):
        c = self.cfg
        ids = torch.tensor(ids, device=self.embed.device)
        pos = torch.arange(len(ids), device=ids.device)
        h = self.embed[ids]
        ni, nr = c["n_input_layers"], c["n_recurrent_layers"]
        for L in self.layers[:ni]:
            h = self.layer(L, h, pos)
        inp = h
        state = inp
        for _ in range(self.loops):
            h = justnorm(state + inp)
            for L in self.layers[ni : ni + nr]:
                h = self.layer(L, h, pos)
            state = justnorm(h)
        h = state
        for L in self.layers[ni + nr :]:
            h = self.layer(L, h, pos)
        logits = (h @ self.head.T).float() * (self.logit_scale / self.std)
        return logits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", required=True, help="comma separated ids, or @file with one id per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loops", type=int, default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    a = ap.parse_args()
    if a.tokens.startswith("@"):
        ids = [int(x) for x in open(a.tokens[1:]).read().split()]
    else:
        ids = [int(x) for x in a.tokens.split(",")]
    m = Ref(a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16, loops=a.loops)
    lg = m.forward(ids)
    torch.save({"ids": ids, "logits": lg.cpu()}, a.out)
    top = lg.argmax(-1).tolist()
    print("argmax", top[-16:])
