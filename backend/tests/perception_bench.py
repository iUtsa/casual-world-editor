"""Perception accuracy against ground truth (synthetic scenes with exact object sizes, depth and id masks).

    python backend/tests/perception_bench.py [--mono-only|--gt-only]   (needs the server on :8000)

For every ground-truth object: was it found (mask IoU >= 0.5), is the shape right (box / cylinder), how far off
are its height and footprint, and the distances between object centres (frame-independent placement check).
Depth modes: "gt" = the renderer's depth (what an RGB-D / LiDAR camera gives), "mono" = the monocular estimate.
"""
import io
import json
import sys
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENES = ["bridge", "tower"]
API = "http://localhost:8000"


def post_scan(rgb_png: bytes, depth_png: bytes | None, hfov: float) -> dict:
    b = uuid.uuid4().hex
    parts = []

    def field(name, value):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

    def file(name, data, fn):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\nContent-Type: image/png\r\n\r\n'.encode() + data + b"\r\n")
    file("image", rgb_png, "rgb.png")
    if depth_png is not None:
        file("depth", depth_png, "depth.png")
    field("hfov", f"{hfov}")
    field("source", "bench")
    field("use_claude", "false")
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    req = urllib.request.Request(f"{API}/api/scan", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def gt_dims(o):
    if o["shape"] == "cylinder":
        d, h = o["size"]
        return sorted([d, d]), h
    sx, sy, sz = o["size"]
    return sorted([sx, sy]), sz


def run(scene_name: str, mode: str):
    d = ROOT / "data" / "synthetic" / scene_name
    meta = json.loads((d / "meta.json").read_text())
    rgb = cv2.imread(str(d / "rgb.png"))
    ids = cv2.imread(str(d / "ids_gt.png"), cv2.IMREAD_UNCHANGED)
    dep = None
    if mode == "gt":
        z = np.load(d / "depth_gt.npy")
        dep = cv2.imencode(".png", np.nan_to_num(z * 1000).clip(0, 65535).astype(np.uint16))[1].tobytes()
    payload = post_scan(cv2.imencode(".png", rgb)[1].tobytes(), dep, meta["hfov_deg"])
    sc = payload["scene"]
    H, W = ids.shape
    dets = []
    for o in sc["objects"]:
        m = np.zeros((H, W), np.uint8)
        poly = np.array(o["polygon"], np.int32)
        if len(poly) >= 3:
            cv2.fillPoly(m, [poly], 1)
        dets.append((o, m.astype(bool)))
    rows = []
    centres_gt, centres_det = [], []
    for k, g in enumerate(meta["objects"], start=1):
        gm = ids == k
        best, best_iou = None, 0.0
        for o, m in dets:
            iou = (gm & m).sum() / max(1, (gm | m).sum())
            if iou > best_iou:
                best, best_iou = o, iou
        found = best is not None and best_iou >= 0.5
        row = {"name": g["name"], "iou": best_iou, "found": found, "scene": scene_name}
        if found:
            gxy, gh = gt_dims(g)
            dxy = sorted(best["size"][:2])
            row.update(label=best["label"], shape_ok=(best["shape"] == g["shape"]), shape=best["shape"],
                       h_err=abs(best["size"][2] - gh) / gh, xy_err=float(np.mean([abs(a - b) / b for a, b in zip(dxy, gxy)])),
                       size=[round(x, 3) for x in best["size"]], gt=[*[round(x, 3) for x in gxy], gh])
            centres_gt.append(np.array(g["pos"], float)); centres_det.append(np.array(best["pos"], float))
        rows.append(row)
    # placement: pairwise centre distances (independent of where each side puts its origin)
    pair_err = []
    for i in range(len(centres_gt)):
        for j in range(i + 1, len(centres_gt)):
            dg = np.linalg.norm(centres_gt[i] - centres_gt[j]); dd = np.linalg.norm(centres_det[i] - centres_det[j])
            pair_err.append(abs(dd - dg) / max(dg, 1e-6))
    extra = len(dets) - sum(r["found"] for r in rows)
    return rows, pair_err, extra, sc


def main():
    modes = ["mono"] if "--mono-only" in sys.argv else ["gt"] if "--gt-only" in sys.argv else ["gt", "mono"]
    summary = {}
    for mode in modes:
        allr, allp, extras = [], [], 0
        for s in SCENES:
            rows, pe, extra, _ = run(s, mode)
            allr += rows; allp += pe; extras += extra
            print(f"\n[{mode} depth] {s}")
            for r in rows:
                if r["found"]:
                    print(f"  {r['name']:11s} found (IoU {r['iou']:.2f})  label '{r['label']}'  shape {r['shape']}{'' if r['shape_ok'] else ' ✗'}"
                          f"  size {r['size']} vs gt {r['gt']}  height err {100 * r['h_err']:.0f}%  footprint err {100 * r['xy_err']:.0f}%")
                else:
                    print(f"  {r['name']:11s} NOT FOUND (best IoU {r['iou']:.2f})")
        f = [r for r in allr if r["found"]]
        # one known measurement ("the mug is 10 cm tall") rescales a scan: errors of all the OTHER objects after that
        cal = []
        for s_ in SCENES:
            rs = [r for r in allr if r["found"] and r.get("scene") == s_]
            if len(rs) >= 2:
                k = rs[0]["gt"][2] / rs[0]["size"][2]
                for r in rs[1:]:
                    cal.append(abs(r["size"][2] * k - r["gt"][2]) / r["gt"][2])
                    cal += [abs(a * k - b) / b for a, b in zip(sorted(r["size"][:2]), r["gt"][:2])]
        summary.setdefault(mode, {})
        summary[mode + " (+1 known size)"] = {"size error after calibrating from one object (median)": f"{100 * np.median(cal):.1f}%" if cal else "-",
                                              "worst": f"{100 * np.max(cal):.0f}%" if cal else "-"}
        summary[mode] = {
            "found": f"{len(f)}/{len(allr)}", "extra (false) objects": extras,
            "shape correct": f"{sum(r['shape_ok'] for r in f)}/{len(f)}",
            "height error (median)": f"{100 * np.median([r['h_err'] for r in f]):.1f}%" if f else "-",
            "footprint error (median)": f"{100 * np.median([r['xy_err'] for r in f]):.1f}%" if f else "-",
            "placement error (median)": f"{100 * np.median(allp):.1f}%" if allp else "-",
        }
    print("\nSUMMARY")
    for mode, sm in summary.items():
        print(f"  {mode:5s}", "  ".join(f"{k}: {v}" for k, v in sm.items()))


if __name__ == "__main__":
    main()
