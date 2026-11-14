"""Quantum thermochemistry for molecules without measured data: H(298 K) = E_DFT + ZPE + thermal (RRHO).

  * geometry: GFN2-xTB optimised (chem.build)
  * electronic energy: DFT single point (PySCF), spin from the structure (O2 triplet, radicals doublet)
  * vibrations: numerical GFN2-xTB Hessian at the same geometry (a stationary point of that method), translations and
    rotations projected out; zero-point energy + vibrational, rotational, translational heat + PV = RT
Everything is cached per (SMILES, method) in data/thermo_cache.json, so a molecule is computed once.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np

from . import config

BOHR = 0.52917721
HARTREE_KJ = 2625.4996
AMU_ME = 1822.888486
CM = 219474.6313705            # hartree -> cm^-1
KB_H = 3.166811563e-6          # hartree / K
T0 = 298.15
METHOD = ("M062X", "def2-TZVP")
# measured on 10 reactions with known gas-phase enthalpies (backend/tests/test_chem_thermo.py::test_dft_benchmark)
BENCH_MAE_KJ = 21.6
BENCH_REL = 0.045             # combustions in the benchmark were 3.6–4.4 % off: errors grow with the size of the reaction
METHOD_TEXT = "M06-2X/def2-TZVP energies on GFN2-xTB geometries + GFN2-xTB vibrations (ZPE, heat to 298 K)"
_CACHE = Path(getattr(config, "ROOT", Path("."))) / "data" / "thermo_cache.json"
_lock = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except Exception:
        return {}


def _save(c: dict) -> None:
    import os
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(c, indent=1))
    os.replace(tmp, _CACHE)


def xtb_hessian(numbers: np.ndarray, pos_bohr: np.ndarray, uhf: int, h: float = 0.005) -> np.ndarray:
    from tblite.interface import Calculator
    n = len(numbers)
    H = np.zeros((3 * n, 3 * n))

    def grad(p):
        c = Calculator("GFN2-xTB", numbers, p, uhf=uhf)
        c.set("verbosity", 0)
        return c.singlepoint().get("gradient").reshape(-1)
    for k in range(3 * n):
        dp = np.zeros(3 * n); dp[k] = h
        gp = grad(pos_bohr + dp.reshape(n, 3)); gm = grad(pos_bohr - dp.reshape(n, 3))
        H[k] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


def frequencies(numbers: np.ndarray, pos_bohr: np.ndarray, H: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, bool]:
    """Harmonic wavenumbers (cm^-1) with translations/rotations projected out; and whether the molecule is linear."""
    n = len(numbers)
    m = np.repeat(masses, 3)
    Hm = H / np.sqrt(np.outer(m, m)) / AMU_ME
    com = (masses[:, None] * pos_bohr).sum(0) / masses.sum()
    r = pos_bohr - com
    I = sum(mm * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri)) for mm, ri in zip(masses, r))
    linear = n == 2 or np.linalg.eigvalsh(I)[0] < 1e-3 * max(1.0, np.linalg.eigvalsh(I)[-1])
    vecs = []
    sq = np.sqrt(masses)
    for ax in range(3):
        v = np.zeros((n, 3)); v[:, ax] = sq; vecs.append(v.reshape(-1))
    for ax in range(3):
        e = np.zeros(3); e[ax] = 1
        v = np.cross(e, r) * sq[:, None]; vecs.append(v.reshape(-1))
    B = np.array(vecs).T
    Q, s, _ = np.linalg.svd(B, full_matrices=False)
    Q = Q[:, s > 1e-6 * s.max()]
    P = np.eye(3 * n) - Q @ Q.T
    w = np.linalg.eigvalsh(P @ Hm @ P)
    w = np.sort(w)[Q.shape[1]:]                              # drop the 5/6 projected (zero) modes
    return np.sign(w) * np.sqrt(np.abs(w)) * CM, bool(linear)


def thermal(freqs_cm: np.ndarray, linear: bool, n_atoms: int, T: float = T0) -> dict:
    kT = KB_H * T
    real = freqs_cm[freqs_cm > 20.0]                         # imaginary / near-zero modes don't store heat
    e = real / CM
    zpe = 0.5 * e.sum()
    evib = float(np.sum(e / np.expm1(e / kT)))
    erot = 0.0 if n_atoms == 1 else (kT if linear else 1.5 * kT)
    return {"zpe": float(zpe), "h_corr": float(zpe + evib + erot + 1.5 * kT + kT), "n_imag": int(np.sum(freqs_cm < -20.0))}


OPTIMISE = False      # benchmarked: DFT-optimised geometries were no better (MAE 25 vs 22 kJ/mol) and 20x slower


def _scf(symbols, pos_ang, spin, charge=0, dm0=None):
    from pyscf import dft, gto
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(symbols, pos_ang)], basis=METHOD[1], spin=spin, charge=charge, verbose=0, unit="Angstrom")
    mf = (dft.UKS(mol) if spin else dft.RKS(mol))
    mf.xc = METHOD[0]
    mf.conv_tol = 1e-9
    mf.grids.level = 4
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        e = mf.kernel()
    return float(e), mf


def dft_energy(symbols: list[str], pos_ang: np.ndarray, spin: int, charge: int = 0) -> tuple[float, np.ndarray]:
    """DFT energy at the DFT minimum: BFGS on analytic DFT forces, starting from the xTB structure."""
    e, mf = _scf(symbols, pos_ang, spin, charge)
    if not OPTIMISE or len(symbols) < 2:
        return e, pos_ang
    from scipy.optimize import minimize
    state = {"dm": mf.make_rdm1()}

    def f(x):
        e_, mf_ = _scf(symbols, x.reshape(-1, 3), spin, charge, state["dm"])
        state["dm"] = mf_.make_rdm1()
        g = mf_.nuc_grad_method().kernel() / BOHR              # hartree/bohr -> hartree/Angstrom
        return e_, g.reshape(-1)
    r = minimize(f, pos_ang.reshape(-1), jac=True, method="BFGS", options={"gtol": 3e-4, "maxiter": 60})
    return float(r.fun), r.x.reshape(-1, 3)


def enthalpy(sp) -> dict:
    """H(298.15 K) in hartree for a built chem.Species (xTB geometry), computed once and cached."""
    from . import chem
    key = f"{sp.smiles}|{METHOD[0]}/{METHOD[1]}|{'opt' if OPTIMISE else 'sp'}"
    with _lock:
        c = _load()
        if key in c:
            return c[key]
    uhf = chem._uhf(sp.smiles)
    e, pos_dft = dft_energy([chem._sym(z) for z in sp.numbers], sp.positions, uhf)
    pos_b = sp.positions / BOHR                             # vibrations: xTB Hessian at the xTB minimum (a stationary point)
    H = xtb_hessian(sp.numbers, pos_b, uhf)
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    masses = np.array([pt.GetMostCommonIsotopeMass(int(z)) for z in sp.numbers])
    freqs, linear = frequencies(sp.numbers, pos_b, H, masses)
    th = thermal(freqs, linear, len(sp.numbers))
    out = {"E": e, "H": e + th["h_corr"], "zpe": th["zpe"], "h_corr": th["h_corr"], "n_imag": th["n_imag"],
           "method": METHOD_TEXT}
    with _lock:
        c = _load(); c[key] = out; _save(c)
    return out


# ---------------------------------------------------------------- background jobs (a molecule takes seconds to minutes)
# PySCF brings GNU OpenMP; the server already holds PyTorch's LLVM OpenMP. Two OpenMP runtimes in one process deadlock
# (measured: server stuck at 0 % CPU in gomp/kmp barriers), so quantum chemistry always runs in its own process.
JOBS: dict[str, dict] = {}


def compute(text: str, progress=lambda k, n: None) -> dict:
    from . import chem
    reactants, products = chem.parse(text)
    coeffs = chem.balance(reactants, products)
    allsp = reactants + products
    nu = [-c for c in coeffs[:len(reactants)]] + list(coeffs[len(reactants):])
    Hs = []
    for k, sp in enumerate(allsp):
        chem.build(sp)
        Hs.append(enthalpy(sp))
        progress(k + 1, len(allsp))
    dh = sum(v * h["H"] for v, h in zip(nu, Hs)) * HARTREE_KJ
    meas = sum(v * s.dhf_gas for v, s in zip(nu, allsp)) if all(s.dhf_gas is not None for s in allsp) else None
    return {"dH_gas_kJmol": round(dh, 1), "measured_gas_kJmol": round(meas, 1) if meas is not None else None,
            "error_kJmol": round(dh - meas, 1) if meas is not None else None,
            "uncertainty_kJmol": round(max(BENCH_MAE_KJ, BENCH_REL * abs(dh)), 1), "method": METHOD_TEXT,
            "imaginary": [s.name for s, h in zip(allsp, Hs) if h["n_imag"]]}


def start(text: str) -> dict:
    import os
    import subprocess
    import sys
    import uuid
    from . import chem
    reactants, products = chem.parse(text)              # fail fast (unknown species, can't balance) before spawning
    chem.balance(reactants, products)
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "state": "running", "done": 0, "total": len(reactants) + len(products)}
    root = Path(__file__).resolve().parents[2]

    def run():
        try:
            env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
            p = subprocess.Popen([sys.executable, "-m", "backend.app.thermo_dft", text], cwd=root, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            result = None
            for line in p.stdout:                       # progress and result arrive as JSON lines
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "done" in msg:
                    JOBS[jid]["done"] = msg["done"]
                if "result" in msg:
                    result = msg["result"]
            p.wait(timeout=60)
            if result is None:
                raise RuntimeError((p.stderr.read() or "quantum chemistry stopped").strip().splitlines()[-1][:300])
            JOBS[jid].update(state="done", done=JOBS[jid]["total"], **result)
        except Exception as e:                          # never leave the page waiting
            JOBS[jid].update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()
    return JOBS[jid]


if __name__ == "__main__":                              # the worker process: python -m backend.app.thermo_dft "<reaction>"
    import sys
    out = compute(sys.argv[1], lambda k, n: print(json.dumps({"done": k, "total": n}), flush=True))
    print(json.dumps({"result": out}), flush=True)
