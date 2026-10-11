"""GPU-free render progress protocol shared by model and native pipelines."""
import json
import math
import sys
import time


def frame_total(duration, fps):
    """Output cadence, including the final source-frame display interval."""
    if not math.isfinite(duration) or not math.isfinite(fps) or duration <= 0 or fps <= 0:
        return 0
    return max(1, math.ceil(duration * fps - 1e-6))


class RenderProgress:
    def __init__(self, total=0, *, clock=time.monotonic, emit=None):
        self.total = max(0, int(total))
        self.clock = clock
        self.emit = emit or (lambda line: print(line, file=sys.stderr, flush=True))
        self.done = 0
        self.rate = 0.0
        self.last_sample = None
        self.last_emit = None

    def _send(self, stage):
        self.emit('<PROGRESS>' + json.dumps(dict(done=self.done, total=self.total,
                    throughput=self.rate, stage=stage), allow_nan=False))
        self.last_emit = self.clock()

    def preparing(self):
        self._send('preparing')

    def rendering(self):
        self._send('rendering')

    def update(self, done):
        now = self.clock()
        # First frame includes decode/model warmup: establish a baseline, not a rate.
        if self.last_sample is not None:
            previous_done, previous_time = self.last_sample
            if done > previous_done and now > previous_time:
                sample = (done - previous_done) / (now - previous_time)
                # Time-weighted EWMA (5 second half-life), independent of frame cadence.
                alpha = 1 - math.exp(-math.log(2) * (now - previous_time) / 5)
                self.rate = sample if self.rate == 0 else self.rate + alpha * (sample - self.rate)
        self.last_sample = (done, now)
        self.done = max(self.done, int(done))
        self.total = max(self.total, self.done)
        if self.last_emit is None or now - self.last_emit >= 1 or self.done == 1 or self.done == self.total:
            self._send('rendering')

    def finalizing(self, done):
        self.done = self.total = int(done)  # Decoder's actual count supersedes metadata.
        self._send('finalizing')
