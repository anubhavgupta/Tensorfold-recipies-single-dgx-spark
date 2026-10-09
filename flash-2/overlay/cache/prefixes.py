"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy."""

import hashlib
import os

KEEP_MAX = int(os.environ.get("TENSORFOLD_KEEP_MAX", "64"))
HEADROOM = int(float(os.environ.get("TENSORFOLD_KEEP_HEADROOM_GIB", "4")) * 2**30)


def keyed(s) -> list[int]:
    """The prompt as a prefix-cache key: each image's placeholder run becomes a token only that image's bytes make."""

    ids, v = list(s.prompt), s.vision
    if v is None:
        return ids

    def tag(kind, h):
        return -1 - int.from_bytes(hashlib.sha256(f"{kind}:{h}".encode()).digest()[:16], "big")

    for (a, b), h in zip(v.image_spans, v.image_hashes):
        ids[a:b] = [tag("i", h)] * (b - a)
    spans, grids = list(getattr(v, "video_spans", ())), getattr(v, "video_grid_thw", None)
    first = 0
    for h, grid in zip(getattr(v, "video_hashes", ()), () if grids is None else grids):
        for a, b in spans[first:first + int(grid[0])]:       # a video is one span a frame group
            ids[a:b] = [tag("v", h)] * (b - a)
        first += int(grid[0])
    return ids


def _best(kept, prompt, busy=()):
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def slot_for(owner, prompt: list[int], reuse: bool):
    """Copy a fork into spare capacity; otherwise retain the released idle-slot and memory-pressure behavior."""

    busy = owner._busy()
    best = _best(owner.kept, prompt) if reuse else None
    fork = best is not None and (id(best[1]) in busy or _longer(owner.kept, best))
    if fork:
        if owner.free:
            spare = owner.free.pop()
            try:
                if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                    spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"])
                    return spare, {"state": best[2], "tail": best[3]}, len(best[0])
            except Exception:
                owner.free.append(spare)
                owner._shrink(spare, force=True)
                raise
            owner.free.append(spare)
        best = _best(owner.kept, prompt, busy) if reuse else None
        if best is not None and owner.free and _longer(owner.kept, best):
            best = None
    if best is not None:
        n = len(best[0])
        owner.kept = [k for k in owner.kept if k[1] is not best[1]
                      or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
        return best[1], {"state": best[2], "tail": best[3]}, n
    if not owner.free:
        idle = next((k[1] for k in owner.kept if id(k[1]) not in busy), None)
        if idle is None:
            raise RuntimeError("no free stream slot")
        owner._drop_kept(idle)
        owner.free.append(idle)
    return owner.free.pop(), None, 0


def remember(owner, ids, st, snap, tail) -> None:
    """Keep each slot's prefix chain, returning displaced idle slots to the free list."""

    gone = [k[1] for k in owner.kept if k[0] == ids]
    owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    while len(owner.kept) > owner.keep:
        gone.append(owner.kept.pop(0)[1])
    while low(owner) and shed(owner):
        pass
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)


def low(owner) -> bool:
    """Whether free memory is under the headroom that extra snapshots may not eat into."""

    live = owner.memory_gate.live
    return live is not None and live() < HEADROOM


def shed(owner, keep=None, protect=None) -> bool:
    """Drop the oldest snapshot beyond the guaranteed ``keep_min`` whose slot keeps another one; False if none."""

    if len(owner.kept) <= owner.keep_min:
        return False
    for k in owner.kept:
        if k[1] is not keep and k[1] is not protect and sum(x[1] is k[1] for x in owner.kept) > 1:
            owner.kept = [x for x in owner.kept if x is not k]
            return True
    return False
