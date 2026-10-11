import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_progress import RenderProgress, frame_total


def test_capped_preview_total():
    assert frame_total(min(5, 90), 60) == 300
    assert frame_total(min(5, 1.2), 24) == 29
    assert frame_total(0, 60) == 0
    assert frame_total(float('nan'), 60) == 0


def test_delayed_start_recent_rate_and_finalizing():
    now = [0.0]
    lines = []
    p = RenderProgress(120, clock=lambda: now[0], emit=lines.append)
    p.preparing()
    now[0] = 600  # compilation and queue waiting cannot pollute render rate
    p.rendering()
    now[0] = 700  # first frame / warmup likewise establishes only a baseline
    p.update(1)
    assert p.rate == 0
    now[0] = 701
    p.update(2)
    assert p.rate == 1
    now[0] = 706
    p.update(12)
    assert p.rate == 1.5  # five-second half-life, new sample 2 FPS
    now[0] = 711
    p.update(13)
    assert abs(p.rate - 0.85) < 1e-9  # slow-down also adjusts
    p.finalizing(14)
    records = [json.loads(line.removeprefix('<PROGRESS>')) for line in lines]
    assert records[0]['stage'] == 'preparing'
    assert records[-1] == dict(done=14, total=14, throughput=p.rate, stage='finalizing')
    assert all(r['done'] <= r['total'] for r in records)


def test_progress_is_time_throttled_not_ten_frame_blocked():
    now = [0.0]
    lines = []
    p = RenderProgress(3, clock=lambda: now[0], emit=lines.append)
    p.rendering()
    now[0] = 50
    p.update(1)
    now[0] = 100
    p.update(2)
    now[0] = 150
    p.update(3)
    assert len(lines) == 4
    assert p.rate == 0.02
