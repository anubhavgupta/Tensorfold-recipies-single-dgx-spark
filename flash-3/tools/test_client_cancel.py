"""patches/0007: requests whose clients leave stop while prefilling, queued or decoding (CPU, fake decoder)."""
import threading, time
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.families.qwen4_exp.cuda.multi_fill import PromptPasses

class St: pass
class E:
    def __init__(self): self.stops = []
ROWS = 16

class FakeDecoder(PromptPasses):
    """Prompts fill one token a pass (20 ms); _fill loops passes as MultiDecoder's does when nothing decodes."""
    def __init__(self):
        self.streams, self.filling, self.fills, self.passed = {}, [], {}, {}
        self.link = self.follower = None; self.fill_yield = False
        self.arrived = lambda: False; self.next_id = 0; self.passes = 0; self.freed = []; self.remembered = []
    def live(self): return len(self.streams) + len(self.filling)
    def admit(self, s):
        s.sid, s.st = self.next_id, St(); self.next_id += 1
        s.st.image_rows = (); s.st.image_features = None
        self.fills[s.sid] = [E(), False, 0, None]; self.filling.append(s)
    def _pass_rows(self): return ROWS
    def _pass(self):
        """ROWS tokens a pass, ending at a kept point as _pieces does; a point reached is remembered (as _joined)."""
        time.sleep(0.02); self.passes += 1
        s = self.filling[0]; e, _, a, _ = self.fills[s.sid]
        b = min(a + ROWS, next((p for p in e.stops if p > a), len(s.prompt)), len(s.prompt))
        self.fills[s.sid][2] = b
        if b in e.stops: self.remembered.append(b)
        if b == len(s.prompt):
            self.filling.remove(s); self.fills.pop(s.sid); self.streams[s.sid] = s; s.take([7])
            return [s] if s.done else []
        return []
    def round(self):
        ended = self._fill() if self.filling else []
        for s in [x for x in self.streams.values() if not x.done]:
            time.sleep(0.01); s.take([1])
            if s.done: ended.append(s)
        return ended
    def finish(self, done):
        for s in done: self.streams.pop(s.sid, None); self.freed.append(s.sid)
    def drop(self): return []

def run(sched, n, count, cancel_at=None, res=None, key=None):
    t0 = time.perf_counter()
    gone = (lambda: time.perf_counter() - t0 > cancel_at) if cancel_at is not None else None
    try:
        out = sched.submit(list(range(n)), count, None, True, (lambda new: bool(gone and gone())), cancelled=gone)
        res[key] = ("done", out, time.perf_counter() - t0)
    except RequestCancelled as e:
        res[key] = ("cancelled", str(e), time.perf_counter() - t0)

d = FakeDecoder(); sched = Scheduler(d, max_streams=1); res = {}
# 1. an 8000-token prompt (10 s of passes) whose client leaves at 0.3 s: filled on to a kept point 256 tokens ahead
run(sched, 8000, 5, 0.3, res, "fill"); print("fill:", res["fill"][0], round(res["fill"][2], 2), "passes", d.passes,
                                             "remembered", d.remembered, "freed", d.freed)
assert res["fill"][0] == "cancelled" and res["fill"][2] < 1.2 and d.passes < 60 and d.freed == [0]
assert len(d.remembered) == 1 and d.remembered[0] >= 256
# 1b. the kept point never lands inside an image, nor within 256 tokens of the prompt end
class V: image_spans = [(800, 1500)]
d._pass_rows = lambda: 1000                       # a pass point 1000 tokens ahead
s = Stream(list(range(2000)), 5, None); s.sid = 99; s.vision = V(); d.fills[99] = [E(), False, 200, None]
assert d._park(s) == 800 and d.fills[99][0].stops == [800]
V.image_spans = [(300, 1500)]; d.fills[99] = [E(), False, 200, None]
assert d._park(s) is None                         # its image starts too close to resume from
s.vision = None; d.fills[99] = [E(), False, 1600, None]
assert d._park(s) is None                         # the prompt end comes first: it finishes and keeps its end
del d.fills[99]; del d._pass_rows
# 2. one lane busy with a 3000-token prompt; a queued request leaves at 0.3 s, the first completes
a = threading.Thread(target=run, args=(sched, 3000, 3, None, res, "busy")); a.start(); time.sleep(0.05)
run(sched, 10, 3, 0.3, res, "queued"); print("queued:", res["queued"][0], round(res["queued"][2], 2))
assert res["queued"][0] == "cancelled" and res["queued"][2] < 1.0
a.join(); print("busy:", res["busy"][0]); assert res["busy"][0] == "done"
# 3. a decoding request whose client leaves mid-reply
run(sched, 5, 10_000, 0.5, res, "decode"); print("decode:", res["decode"][0], round(res["decode"][2], 2))
assert res["decode"][0] == "cancelled" and res["decode"][2] < 1.5
# 4. no cancel callable: stock behaviour
run(sched, 5, 4, None, res, "plain"); print("plain:", res["plain"][0]); assert res["plain"][0] == "done"
assert d.live() == 0
print("OK")
