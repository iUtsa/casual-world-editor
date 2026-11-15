"""Where do a live hand session and its replay diverge? Same recorded hand, the scanned bridge scene."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np  # noqa: E402
from backend.app import synth  # noqa: E402
from backend.app.perception import pipeline  # noqa: E402
from backend.app.physics.hand import HandSession  # noqa: E402
from backend.app.physics.sim import simulate  # noqa: E402
from backend.app.schemas import Intervention  # noqa: E402
from test_sweep import _flat_hand  # noqa: E402
info = synth.render("bridge")
rgb = cv2.cvtColor(cv2.imread(info["rgb"]), cv2.COLOR_BGR2RGB)
st = pipeline.scan(rgb, info["hfov_deg"], "s", use_claude=False, depth_override=np.load(info["rgb"].replace("rgb.png", "depth_gt.npy")), label=False)
o = min(st.scene.objects, key=lambda x: x.pos[0]); x0, y0 = o.pos[0] - max(o.size[:2]) / 2 - 0.08, o.pos[1]
hs = HandSession(st.scene, _flat_hand(x0, y0, 0.015), [], st.baseline_qpos, [], 9.81)
t = 0.0; live = {}
while t < 2.8:
    J = _flat_hand(x0 + 0.35 * min(t, 0.9), y0, 0.015).tolist() if t < 1.3 else None
    hs.update(J, t); live[round(hs.t, 3)] = hs.poses()[o.id][:3]; t += 1 / 30
sim = simulate(st.scene, st.baseline_qpos, [Intervention(type="hand", target=None, params=hs.intervention_params())])
i = sim["object_ids"].index(o.id); fps = sim.get("fps", 60)
for tt in (0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.4):
    lk = min(live, key=lambda k: abs(k - tt)); f = min(len(sim["frames"]) - 1, int(round(tt * fps)))
    print(f"t={tt:.1f}  live {np.round(live[lk], 4)}  replay {np.round(sim['frames'][f][i][:3], 4)}")
print("replay frames", len(sim["frames"]), "fps", fps, "live sim time", round(hs.t, 3))

print("--- determinism and pre-contact")
sim2 = simulate(st.scene, st.baseline_qpos, [Intervention(type="hand", target=None, params=hs.intervention_params())])
print("replay twice identical:", sim2["frames"][-1][i][:3] == sim["frames"][-1][i][:3])
hs2 = HandSession(st.scene, _flat_hand(x0, y0, 0.015), [], st.baseline_qpos, [], 9.81)
t = 0.0; live2 = {}
while t < 2.8:
    J = _flat_hand(x0 + 0.35 * min(t, 0.9), y0, 0.015).tolist() if t < 1.3 else None
    hs2.update(J, t); live2[round(hs2.t, 3)] = hs2.poses()[o.id][:3]; t += 1 / 30
print("live twice identical:", live2[max(live2)] == live[max(live)])
for tt in (0.0, 0.05, 0.1, 0.2, 0.25):
    lk = min(live, key=lambda k: abs(k - tt)); f = min(len(sim["frames"]) - 1, int(round(tt * fps)))
    print(f"t={tt:.2f} live {np.round(live[lk], 5)} replay {np.round(sim['frames'][f][i][:3], 5)}")
print("baseline block pose:", np.round(st.baseline_qpos[0:3], 5) if False else "", "scene pos", o.pos)

print("--- equal-time comparison (both at 60 Hz)")
rec = hs.record
first_bad = None
for f in range(min(len(rec), len(sim["frames"]))):
    lp = np.array(rec[f]["poses"][o.id][:3]); rp = np.array(sim["frames"][f][i][:3]); d = np.linalg.norm(lp - rp)
    if first_bad is None and d > 0.0005:
        first_bad = f
    if f % 15 == 0 or f == first_bad:
        print(f"frame {f:3d} t={rec[f]['t']:.3f}/{f / fps:.3f}  diff {d * 1000:6.2f} mm")
print("first frame differing > 0.5 mm:", first_bad, "| live record len", len(rec), "replay frames", len(sim["frames"]))
