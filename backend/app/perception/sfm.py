"""Where the phone was for every frame of a hand-held sweep, from the video alone (iPhone Safari gives no tracking).

  1. SIFT features in every keyframe, matched between nearby frames (and across the sweep, so coming back to the
     start closes the loop); each match set is checked with the epipolar geometry (essential matrix, RANSAC).
  2. Matches are chained into tracks: one physical point seen in many frames.
  3. Incremental registration: the reference frame's metric depth (Depth Anything V2) puts its tracks in 3D;
     every next frame is placed by PnP on the tracks it shares with the frames already placed, its depth scale is
     measured on those same points, and its new tracks are added.
  4. Bundle adjustment: every camera pose, every frame's depth scale and every 3D point are refined together so
     the points reproject onto their pixels (sub-pixel) and agree with the depth maps; the depth maps only fix the
     overall metric scale (their mean log-scale is held at zero), the geometry comes from the pixels.

Poses are returned as (R_wc, t_wc): world-from-camera rotation and camera centre, OpenCV camera axes, with the
world = the reference camera's frame. Units: metres.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SfmResult:
    poses: list[tuple[np.ndarray, np.ndarray]]      # (R_wc, t_wc) per frame
    scales: np.ndarray                              # (unused: kept 1) per-frame depth scale
    points: np.ndarray                              # (M, 3) track points in the world
    obs: list[tuple[int, int, float, float]]        # (frame, point, u, v)
    rms_px: float
    registered: list[bool]
    notes: list[str] = field(default_factory=list)
    depths: list = field(default_factory=list)       # depth maps calibrated to the triangulated points (metres)
    depth_err: list = field(default_factory=list)    # each map's median relative error on those points
    metric_scale: float = 1.0


def _features(gray: np.ndarray, n: int = 5000):
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=n, contrastThreshold=0.015)
    kp, d = sift.detectAndCompute(g, None)
    if d is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kp]), d


def _match(d1, d2, ratio: float = 0.8) -> np.ndarray:
    if len(d1) < 8 or len(d2) < 8:
        return np.zeros((0, 2), int)
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    a = fl.knnMatch(d1, d2, k=2)
    b = fl.knnMatch(d2, d1, k=2)
    back = {m[0].queryIdx: m[0].trainIdx for m in b if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    out = [(m[0].queryIdx, m[0].trainIdx) for m in a
           if len(m) == 2 and m[0].distance < ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
    return np.array(out, int).reshape(-1, 2)


def _verify(p1, p2, K) -> np.ndarray:
    if len(p1) < 15:
        return np.zeros(len(p1), bool)
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        return np.zeros(len(p1), bool)
    return mask.ravel().astype(bool)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _pairs(n: int) -> list[tuple[int, int]]:
    s = set()
    for i in range(n):
        for g in (1, 2, 3, 4):
            if i + g < n:
                s.add((i, i + g))
    for i in range(0, n, 3):                      # a sparse all-pairs net: closes loops when the sweep comes back
        for j in range(i + 6, n, 3):
            s.add((i, j))
    return sorted(s)


def _rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, np.float64).reshape(3, 1))[0]


def _sample(D: np.ndarray, u: np.ndarray, v: np.ndarray, hw) -> np.ndarray:
    h, w = hw
    dh, dw = D.shape
    return D[np.clip((v * dh / h).astype(int), 0, dh - 1), np.clip((u * dw / w).astype(int), 0, dw - 1)]


def reconstruct(frames: list[np.ndarray], depths: list[np.ndarray], K: np.ndarray, ref: int = 0,
                progress=lambda msg: None) -> SfmResult:
    n = len(frames)
    hw = frames[0].shape[:2]
    feats = [_features(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)) for f in frames]
    progress(f"features: {sum(len(f[0]) for f in feats)} points in {n} frames")
    uf = _UF()
    n_pairs = 0
    for i, j in _pairs(n):
        m = _match(feats[i][1], feats[j][1])
        if len(m) < 15:
            continue
        ok = _verify(feats[i][0][m[:, 0]], feats[j][0][m[:, 1]], K)
        if ok.sum() < 20:
            continue
        n_pairs += 1
        for a, b in m[ok]:
            uf.union((i, int(a)), (j, int(b)))
    groups: dict = {}
    for key in list(uf.p.keys()):
        groups.setdefault(uf.find(key), []).append(key)
    tracks = []
    for g in groups.values():
        # a chain of matches can pick up two keypoints in one frame; drop just that frame from the track
        cnt: dict = {}
        for f, _ in g:
            cnt[f] = cnt.get(f, 0) + 1
        g = [(f, k) for f, k in g if cnt[f] == 1]
        if len(g) >= 2:
            tracks.append(sorted(g))
    progress(f"{len(tracks)} tracks from {n_pairs} matched frame pairs")
    by_frame: list[dict[int, int]] = [dict() for _ in range(n)]   # frame -> {keypoint: track}
    for t, g in enumerate(tracks):
        for f, k in g:
            by_frame[f][k] = t

    # ---- incremental registration
    R_cw = [None] * n                 # camera-from-world rotation
    t_cw = [None] * n
    scale = np.ones(n)
    X = np.full((len(tracks), 3), np.nan)
    R_cw[ref], t_cw[ref] = np.eye(3), np.zeros(3)

    def add_points(f):
        Kinv = np.linalg.inv(K)
        for k, t in by_frame[f].items():
            if np.isfinite(X[t, 0]):
                continue
            u, v = feats[f][0][k]
            d = scale[f] * float(_sample(depths[f], np.array([u]), np.array([v]), hw)[0])
            if not (0.05 < d < 20):
                continue
            pc = Kinv @ np.array([u, v, 1.0]) * d
            X[t] = R_cw[f].T @ (pc - t_cw[f])
    add_points(ref)
    done = {ref}
    while len(done) < n:
        best, best_c = None, []
        for f in range(n):
            if f in done:
                continue
            c = [(k, t) for k, t in by_frame[f].items() if np.isfinite(X[t, 0])]
            if len(c) > len(best_c):
                best, best_c = f, c
        if best is None or len(best_c) < 12:
            break
        f = best
        obj = np.array([X[t] for _, t in best_c])
        img = np.array([feats[f][0][k] for k, _ in best_c], np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=500, reprojectionError=4.0,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < 10:
            done.add(f)                    # give up on this frame; it stays unregistered
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        R_cw[f], t_cw[f] = _rodrigues(rvec), tvec.ravel()
        pc = obj[inl] @ R_cw[f].T + t_cw[f]
        d = _sample(depths[f], img[inl, 0], img[inl, 1], hw)
        good = (pc[:, 2] > 0.05) & (d > 0.05)
        if good.sum() >= 5:
            scale[f] = float(np.median(pc[good, 2] / d[good]))
        done.add(f)
        add_points(f)
    registered = [R is not None for R in R_cw]
    progress(f"registered {sum(registered)}/{n} frames")

    # ---- bundle adjustment (pixels only), then the metric scale from the depth maps
    obs = []
    for f in range(n):
        if not registered[f]:
            continue
        for k, t in by_frame[f].items():
            if np.isfinite(X[t, 0]):
                u, v = feats[f][0][k]
                obs.append((f, t, float(u), float(v)))
    R_cw, t_cw, X, rms, obs = bundle_adjust(R_cw, t_cw, X, obs, K, ref, registered)
    ratios = []
    for f, t, u, v in obs:
        z = (R_cw[f] @ X[t] + t_cw[f])[2]
        d = float(_sample(depths[f], np.array([u]), np.array([v]), hw)[0])
        if z > 0.02 and d > 0.05:
            ratios.append(d / z)
    s_glob = float(np.median(ratios)) if ratios else 1.0
    X = X * s_glob
    t_cw = [t * s_glob if t is not None else None for t in t_cw]
    progress(f"bundle adjustment: {len(obs)} observations, reprojection rms {rms:.2f} px, metric scale x{s_glob:.3f}")
    poses, aligned, depth_err = [], [], []
    for f in range(n):
        if registered[f]:
            poses.append((R_cw[f].T, -R_cw[f].T @ t_cw[f]))
            mine = [(u, v, (R_cw[f] @ X[t] + t_cw[f])[2]) for f2, t, u, v in obs if f2 == f]
            arr = np.array(mine) if mine else np.zeros((0, 3))
            Dc, e = align_depth(depths[f], arr[:, :2], arr[:, 2], hw) if len(arr) else (depths[f], 1.0)
            aligned.append(Dc)
            depth_err.append(e)
        else:
            poses.append((np.eye(3), np.zeros(3)))
            aligned.append(None)
            depth_err.append(1.0)
    r = SfmResult(poses, np.ones(n), X, obs, rms, registered)
    r.depths = aligned
    r.depth_err = depth_err
    r.metric_scale = s_glob
    return r


def _skew_neg(a: np.ndarray) -> np.ndarray:
    """-[a]x for a batch of vectors (the derivative of R X w.r.t. a left rotation perturbation)."""
    S = np.zeros((len(a), 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = a[:, 2], -a[:, 1], -a[:, 2]
    S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = a[:, 0], a[:, 1], -a[:, 0]
    return S


def bundle_adjust(R_cw, t_cw, X, obs, K, ref, registered, iters: int = 30, huber: float = 1.5):
    """Reprojection-only bundle adjustment: Levenberg-Marquardt with the Schur complement (points eliminated, the
    small camera system solved directly) and Huber weights, the standard way (Triggs et al. 2000).

    The reference camera is fixed; the overall scale is gauge-fixed after every step (the camera furthest from the
    reference keeps its distance). Observations still far off at the end are dropped as bad matches and the
    problem is solved once more without them."""
    from scipy.sparse import csr_matrix
    frames = [f for f in range(len(R_cw)) if registered[f] and f != ref]
    fidx = {f: i for i, f in enumerate(frames)}
    pts = sorted({t for _, t, _, _ in obs})
    pidx = {t: i for i, t in enumerate(pts)}
    O = np.array([(f, pidx[t], u, v) for f, t, u, v in obs])
    nF, nP = len(frames), len(pts)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Rc = np.array([R_cw[f] for f in frames]) if nF else np.zeros((0, 3, 3))
    Tc = np.array([t_cw[f] for f in frames]) if nF else np.zeros((0, 3))
    P = X[pts].copy()
    Rref, tref = R_cw[ref], t_cw[ref]
    c_ref = -Rref.T @ tref
    cen = lambda R, t: -R.T @ t
    ai = int(np.argmax([np.linalg.norm(cen(Rc[i], Tc[i]) - c_ref) for i in range(nF)])) if nF else -1
    d0 = float(np.linalg.norm(cen(Rc[ai], Tc[ai]) - c_ref)) if nF else 1.0

    def fix_gauge(Rc, Tc, P):
        if not nF:
            return Rc, Tc, P
        k = d0 / max(1e-9, float(np.linalg.norm(cen(Rc[ai], Tc[ai]) - c_ref)))
        # scale the world about the reference camera centre: X' = c + k (X - c); t' = R c (1 - k) + k t
        P = c_ref + k * (P - c_ref)
        Tc = np.einsum("nij,j->ni", Rc, c_ref) * (1 - k) + k * Tc
        return Rc, Tc, P

    def rot(w):
        return np.array([_rodrigues(x) for x in w])

    rms = 0.0
    keep = np.ones(len(O), bool)
    for stage in range(2):
        Ok = O[keep]
        of, op, ou, ov = Ok[:, 0].astype(int), Ok[:, 1].astype(int), Ok[:, 2], Ok[:, 3]
        ci = np.array([fidx.get(f, -1) for f in of])
        isr = ci < 0
        cc = np.maximum(ci, 0)
        m = len(Ok)

        def project_all(Rc, Tc, P):
            R = np.where(isr[:, None, None], Rref[None], Rc[cc] if nF else Rref[None])
            t = np.where(isr[:, None], tref[None], Tc[cc] if nF else tref[None])
            RX = np.einsum("nij,nj->ni", R, P[op])
            pc = RX + t
            z = np.maximum(pc[:, 2], 1e-4)
            r = np.stack([fx * pc[:, 0] / z + cx - ou, fy * pc[:, 1] / z + cy - ov], 1)
            return r, R, RX, pc, z

        def cost(r):
            e = np.linalg.norm(r, axis=1)
            return float(np.sum(np.where(e < huber, 0.5 * e ** 2, huber * (e - 0.5 * huber))))

        lam = 1e-3
        r, R, RX, pc, z = project_all(Rc, Tc, P)
        c0 = cost(r)
        for it in range(iters):
            e = np.linalg.norm(r, axis=1)
            w = np.where(e < huber, 1.0, huber / np.maximum(e, 1e-9))
            Jpi = np.zeros((m, 2, 3))
            Jpi[:, 0, 0], Jpi[:, 0, 2] = fx / z, -fx * pc[:, 0] / z ** 2
            Jpi[:, 1, 1], Jpi[:, 1, 2] = fy / z, -fy * pc[:, 1] / z ** 2
            Jc = np.concatenate([Jpi @ _skew_neg(RX), Jpi], 2)                      # (m,2,6)
            Jp = Jpi @ R                                                            # (m,2,3)
            Jc[isr] = 0
            JcW = Jc * w[:, None, None]
            JpW = Jp * w[:, None, None]
            U = np.zeros((nF, 6, 6)); np.add.at(U, cc, np.einsum("nki,nkj->nij", JcW, Jc))
            V = np.zeros((nP, 3, 3)); np.add.at(V, op, np.einsum("nki,nkj->nij", JpW, Jp))
            Wb = np.einsum("nki,nkj->nij", JcW, Jp)                                 # (m,6,3)
            gc = np.zeros((nF, 6)); np.add.at(gc, cc, np.einsum("nki,nk->ni", JcW, r))
            gp = np.zeros((nP, 3)); np.add.at(gp, op, np.einsum("nki,nk->ni", JpW, r))
            ok_obs = ~isr
            rows = (6 * cc[ok_obs, None, None] + np.arange(6)[None, :, None]).repeat(3, 2)
            cols = (3 * op[ok_obs, None, None] + np.arange(3)[None, None, :]).repeat(6, 1)
            Wsp = csr_matrix((Wb[ok_obs].ravel(), (rows.ravel(), cols.ravel())), shape=(6 * nF, 3 * nP))
            accepted = False
            for _ in range(8):
                Ud = U + lam * (np.einsum("nii->ni", U)[..., None] * np.eye(6) + 1e-9 * np.eye(6))
                Vd = V + lam * (np.einsum("nii->ni", V)[..., None] * np.eye(3)) + 1e-9 * np.eye(3)
                Vi = np.linalg.inv(Vd)
                rr = (3 * np.arange(nP)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
                cl = (3 * np.arange(nP)[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
                Visp = csr_matrix((Vi.ravel(), (rr.ravel(), cl.ravel())), shape=(3 * nP, 3 * nP))
                WV = Wsp @ Visp
                S = (np.zeros((6 * nF, 6 * nF)) if nF else np.zeros((0, 0)))
                for i in range(nF):
                    S[6 * i:6 * i + 6, 6 * i:6 * i + 6] = Ud[i]
                S -= (WV @ Wsp.T).toarray()
                bc = -gc.ravel() + WV @ gp.ravel()
                try:
                    dc = np.linalg.solve(S, bc) if nF else np.zeros(0)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                dp = Visp @ (-gp.ravel() - Wsp.T @ dc)
                dc = dc.reshape(nF, 6)
                Rn = np.einsum("nij,njk->nik", rot(dc[:, :3]), Rc) if nF else Rc
                Tn = Tc + dc[:, 3:]
                Pn = P + dp.reshape(nP, 3)
                Rn, Tn, Pn = fix_gauge(Rn, Tn, Pn)
                rn, Rn_, RXn, pcn, zn = project_all(Rn, Tn, Pn)
                cn = cost(rn)
                if cn < c0:
                    Rc, Tc, P = Rn, Tn, Pn
                    r, R, RX, pc, z = rn, Rn_, RXn, pcn, zn
                    improvement = (c0 - cn) / max(c0, 1e-12)
                    c0 = cn
                    lam = max(lam / 3, 1e-7)
                    accepted = True
                    break
                lam *= 10
            if not accepted or improvement < 1e-6:
                break
        e = np.linalg.norm(r, axis=1)
        rms = float(np.sqrt(np.mean(e[e < 4.0] ** 2))) if (e < 4.0).any() else float(np.sqrt(np.mean(e ** 2)))
        bad = e > max(2.5, 4 * np.median(e))
        if stage == 1 or not bad.any():
            break
        # a camera that most of its points disagree with has a wrong POSE (bad start), not wrong points:
        # place it again by PnP on the refined points instead of throwing its observations away
        replaced = 0
        for i in range(nF):
            mine = ci == i
            if mine.sum() >= 12 and bad[mine].mean() > 0.4:
                ok_, rv_, tv_, inl_ = cv2.solvePnPRansac(P[op[mine]], np.stack([ou[mine], ov[mine]], 1), K, None,
                                                         iterationsCount=500, reprojectionError=3.0,
                                                         flags=cv2.SOLVEPNP_SQPNP)
                if ok_ and inl_ is not None and len(inl_) >= 10:
                    Rc[i], Tc[i] = _rodrigues(rv_), tv_.ravel()
                    replaced += 1
        if replaced:
            r, R, RX, pc, z = project_all(Rc, Tc, P)
            e = np.linalg.norm(r, axis=1)
            bad = e > max(2.5, 4 * np.median(e))
        idx = np.flatnonzero(keep)
        keep[idx[bad]] = False
    R_out, t_out = list(R_cw), list(t_cw)
    for f, i in fidx.items():
        R_out[f] = Rc[i]
        t_out[f] = Tc[i].copy()
    X_out = X.copy()
    X_out[pts] = P
    kept = [o for o, k in zip(obs, keep) if k]
    return R_out, t_out, X_out, rms, kept


def align_depth(D: np.ndarray, uv: np.ndarray, z: np.ndarray, hw) -> tuple[np.ndarray, float]:
    """Calibrate a depth-model map to triangulated points: 1/z = a/D + b, robust (iteratively reweighted).
    Returns the corrected map and the median relative error on the points (how far to trust it)."""
    d = _sample(D, uv[:, 0], uv[:, 1], hw)
    ok = (d > 0.05) & (z > 0.05)
    if ok.sum() < 20:
        return D, 1.0
    A = np.stack([1 / d[ok], np.ones(ok.sum())], 1)
    y = 1 / z[ok]
    w = np.ones(ok.sum())
    for _ in range(6):
        c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        r = np.abs(A @ c - y) / y
        w = 1 / np.maximum(r, 0.01)
    inv = c[0] / np.maximum(D, 1e-3) + c[1]
    Dc = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf).astype(np.float32)
    zc = 1 / np.maximum(A @ c, 1e-3)
    return Dc, float(np.median(np.abs(zc - z[ok]) / z[ok]))
