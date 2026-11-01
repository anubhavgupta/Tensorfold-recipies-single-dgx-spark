"""patches/0007: requests whose clients leave stop while prefilling, queued or decoding (CPU, fake decoder)."""
import threading, time
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.families.qwen4_exp.cuda.multi_fill import PromptPasses

class St: pass

class FakeDecoder(PromptPasses):
    """Prompts fill one token a pass (20 ms); _fill loops passes as MultiDecoder's does when nothing decodes."""
    def __init__(self):
        self.streams, self.filling, self.fills, self.passed = {}, [], {}, {}
        self.link = self.follower = None; self.fill_yield = False
        self.arrived = lambda: False; self.next_id = 0; self.passes = 0; self.freed = []
    def live(self): return len(self.streams) + len(self.filling)
    def admit(self, s):
        s.sid, s.st = self.next_id, St(); self.next_id += 1
        s.st.image_rows = (); s.st.image_features = None
        self.fills[s.sid] = [None, False, 0, None]; self.filling.append(s)
    def _pass(self):
        time.sleep(0.02); self.passes += 1
        s = self.filling[0]; self.fills[s.sid][2] += 1
        if self.fills[s.sid][2] == len(s.prompt):
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
# 1. a 500-token prompt (10 s of passes) whose client leaves at 0.3 s
run(sched, 500, 5, 0.3, res, "fill"); print("fill:", res["fill"][0], round(res["fill"][2], 2), "passes", d.passes, "freed", d.freed)
assert res["fill"][0] == "cancelled" and res["fill"][2] < 1.0 and d.passes < 50 and d.freed == [0]
# 2. one lane busy with a 100-token prompt; a queued request leaves at 0.3 s, the first completes
a = threading.Thread(target=run, args=(sched, 100, 3, None, res, "busy")); a.start(); time.sleep(0.05)
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
