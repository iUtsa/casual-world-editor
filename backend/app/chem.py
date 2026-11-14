"""Chemistry engine: reaction text -> balanced equation -> 3D molecules -> quantum-driven
reaction dynamics.

  * Species: a curated table of names/formulas -> SMILES (never guessed), or raw SMILES.
  * Balancing: exact integer null space of the element matrix (sympy, rational arithmetic).
  * Geometry: RDKit ETKDG embedding, then GFN2-xTB geometry optimisation (tblite).
  * Energetics: xTB reaction energy, shown next to the tabulated experimental standard
    enthalpy of reaction (from standard enthalpies of formation) where both are known.
  * Dynamics: steered molecular dynamics. Every atom moves under GFN2-xTB quantum forces
    (Langevin, 0.5 fs steps). A slowly increasing bias pulls bonds that must form toward
    their product length and pushes bonds that must break apart - the standard way to see
    a reaction in a simulation that lasts picoseconds. Bonds are tracked by Wiberg bond
    orders computed by xTB at every recorded frame; breaking/forming events are their
    0.5 crossings.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)
BOHR = 0.52917721           # Angstrom per bohr
HARTREE_KJ = 2625.4996      # kJ/mol per hartree
AMU = 1822.888486           # electron masses per amu
FS = 41.341374575751        # atomic time units per femtosecond
KB = 3.166811563e-6         # hartree per kelvin

# name / formula -> (SMILES, ΔHf° in its standard state at 25 °C (kJ/mol), display name, standard phase, ΔHf° as a gas)
# Measured standard enthalpies of formation, 298.15 K, 1 bar (NIST Chemistry WebBook / CRC Handbook). Gas values let every
# reaction also be stated "all gases"; their difference from the condensed value is the measured heat of vaporisation.
SPECIES: dict[str, tuple[str, float | None, str, str, float | None]] = {}
def _sp(keys, smi, dhf, name, phase="g", gas=None):
    for k in keys:
        SPECIES[k.lower()] = (smi, dhf, name, phase, dhf if phase == "g" else gas)
_sp(["h2o", "water"], "O", -285.83, "water", "l", -241.826)
_sp(["h2", "hydrogen", "hydrogen gas"], "[H][H]", 0.0, "hydrogen")
_sp(["o2", "oxygen", "oxygen gas"], "O=O", 0.0, "oxygen")
_sp(["n2", "nitrogen"], "N#N", 0.0, "nitrogen")
_sp(["cl2", "chlorine"], "ClCl", 0.0, "chlorine")
_sp(["o3", "ozone"], "[O-][O+]=O", 142.7, "ozone")
_sp(["ch4", "methane"], "C", -74.6, "methane")
_sp(["c2h6", "ethane"], "CC", -84.0, "ethane")
_sp(["c2h4", "ethene", "ethylene"], "C=C", 52.4, "ethene")
_sp(["c2h2", "ethyne", "acetylene"], "C#C", 227.4, "ethyne")
_sp(["c3h6", "propene", "propylene"], "C=CC", 20.0, "propene")
_sp(["c3h8", "propane"], "CCC", -103.8, "propane")
_sp(["c4h10", "butane"], "CCCC", -125.7, "butane")
_sp(["co2", "carbon dioxide"], "O=C=O", -393.51, "carbon dioxide")
_sp(["co", "carbon monoxide"], "[C-]#[O+]", -110.53, "carbon monoxide")
_sp(["nh3", "ammonia"], "N", -45.9, "ammonia")
_sp(["hcl", "hydrogen chloride", "hydrochloric acid"], "Cl", -92.31, "hydrogen chloride")
_sp(["hf", "hydrogen fluoride"], "F", -273.3, "hydrogen fluoride")
_sp(["hcn", "hydrogen cyanide"], "C#N", 135.1, "hydrogen cyanide")
_sp(["naoh", "sodium hydroxide"], "[Na]O", -425.8, "sodium hydroxide", "s")
_sp(["nacl", "sodium chloride", "salt"], "[Na]Cl", -411.2, "sodium chloride", "s")
_sp(["ch3oh", "ch4o", "methanol"], "CO", -239.2, "methanol", "l", -201.0)
_sp(["c2h5oh", "c2h6o", "ethanol"], "CCO", -277.6, "ethanol", "l", -234.8)
_sp(["c6h12o6", "glucose"], "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", -1273.3, "glucose", "s")
_sp(["h2o2", "hydrogen peroxide"], "OO", -187.8, "hydrogen peroxide", "l", -136.3)
_sp(["c6h6", "benzene"], "c1ccccc1", 49.1, "benzene", "l", 82.9)
_sp(["ch3cooh", "c2h4o2", "acetic acid"], "CC(=O)O", -484.3, "acetic acid", "l", -432.2)
_sp(["hcooh", "ch2o2", "formic acid"], "C(=O)O", -425.0, "formic acid", "l", -378.7)
_sp(["ch2o", "formaldehyde"], "C=O", -108.6, "formaldehyde")
_sp(["ch3cho", "c2h4o", "acetaldehyde"], "CC=O", -166.2, "acetaldehyde")
_sp(["c3h6o", "acetone", "propanone"], "CC(C)=O", -248.4, "acetone", "l", -217.1)
_sp(["ch3cl", "chloromethane", "methyl chloride"], "CCl", -81.9, "chloromethane")
_sp(["ccl4", "carbon tetrachloride", "tetrachloromethane"], "ClC(Cl)(Cl)Cl", -128.2, "carbon tetrachloride", "l", -95.6)
_sp(["no", "nitric oxide", "nitrogen monoxide"], "[N]=O", 91.3, "nitric oxide")
_sp(["no2", "nitrogen dioxide"], "N(=O)[O]", 33.2, "nitrogen dioxide")
_sp(["n2o", "nitrous oxide", "laughing gas"], "[N-]=[N+]=O", 81.6, "nitrous oxide")
_sp(["n2o4", "dinitrogen tetroxide"], "O=[N+]([O-])[N+](=O)[O-]", 11.1, "dinitrogen tetroxide")
_sp(["hno3", "nitric acid"], "O[N+](=O)[O-]", -174.1, "nitric acid", "l", -133.9)
_sp(["so2", "sulfur dioxide"], "O=S=O", -296.8, "sulfur dioxide")
_sp(["so3", "sulfur trioxide"], "O=S(=O)=O", -395.7, "sulfur trioxide")
_sp(["h2so4", "sulfuric acid", "sulphuric acid"], "OS(=O)(=O)O", -814.0, "sulfuric acid", "l", -735.1)
_sp(["h2s", "hydrogen sulfide"], "S", -20.6, "hydrogen sulfide")

TRIPLET = {"O=O"}
DISPLAY = {"O": "H2O", "[H][H]": "H2", "O=O": "O2", "N#N": "N2", "ClCl": "Cl2", "C": "CH4", "CC": "C2H6", "C=C": "C2H4",
           "C#C": "C2H2", "CCC": "C3H8", "CCCC": "C4H10", "O=C=O": "CO2", "[C-]#[O+]": "CO", "N": "NH3", "Cl": "HCl",
           "[Na]O": "NaOH", "[Na]Cl": "NaCl", "CO": "CH3OH", "CCO": "C2H5OH", "OO": "H2O2", "c1ccccc1": "C6H6",
           "CC(=O)O": "CH3COOH", "C(=O)O": "HCOOH", "C=O": "CH2O", "[N]=O": "NO", "N(=O)[O]": "NO2", "O=S=O": "SO2",
           "S": "H2S", "F": "HF", "[O-][O+]=O": "O3", "C=CC": "C3H6", "C#N": "HCN", "CC=O": "CH3CHO", "CC(C)=O": "(CH3)2CO",
           "CCl": "CH3Cl", "ClC(Cl)(Cl)Cl": "CCl4", "[N-]=[N+]=O": "N2O", "O=[N+]([O-])[N+](=O)[O-]": "N2O4", "O[N+](=O)[O-]": "HNO3",
           "O=S(=O)=O": "SO3", "OS(=O)(=O)O": "H2SO4", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O": "C6H12O6"}            # ground-state triplet molecules


@dataclass
class Species:
    token: str
    smiles: str
    name: str
    dhf: float | None
    phase: str = "g"
    dhf_gas: float | None = None
    mol: object = None        # RDKit mol with Hs and a conformer
    energy: float = 0.0       # xTB, hartree
    positions: np.ndarray | None = None   # Angstrom
    numbers: np.ndarray | None = None


def resolve(token: str) -> Species:
    from rdkit import Chem
    t = token.strip()
    key = re.sub(r"\s+", " ", t.lower())
    key = re.sub(r"\((g|l|s|aq)\)$", "", key).strip()
    if key in SPECIES:
        smi, dhf, name, ph, gas = SPECIES[key]
        return Species(t, smi, name, dhf, ph, gas)
    m = Chem.MolFromSmiles(t)
    if m is not None and t not in ("", ):
        can = Chem.MolToSmiles(m)
        for smi, dhf, name, ph, gas in SPECIES.values():
            if Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == can:
                return Species(t, smi, name, dhf, ph, gas)
        return Species(t, can, can, None)
    raise ValueError(f"unknown species '{t}': use a common name/formula ({', '.join(sorted({v[2] for v in SPECIES.values()}))}) or a SMILES string")


def parse(text: str) -> tuple[list[Species], list[Species]]:
    t = text.replace("→", "->").replace("⟶", "->").replace("=>", "->")
    if "->" not in t and " = " in t:
        t = t.replace(" = ", " -> ")
    if "->" not in t:
        raise ValueError("write the reaction with an arrow, e.g. 'methane + oxygen -> carbon dioxide + water'")
    lhs, rhs = t.split("->", 1)

    def side(s):
        out = []
        for part in re.split(r"\s\+\s|\s+and\s+|,", s):
            part = re.sub(r"^\s*\d+\s*", "", part.strip())       # coefficients are recomputed
            if part:
                out.append(resolve(part))
        return out
    return side(lhs), side(rhs)


def _formula_counts(smiles: str) -> dict[str, int]:
    from rdkit import Chem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    c: dict[str, int] = {}
    for a in m.GetAtoms():
        c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
    return c


def balance(reactants: list[Species], products: list[Species]) -> list[int]:
    import sympy
    species = reactants + products
    counts = [_formula_counts(s.smiles) for s in species]
    elements = sorted({e for c in counts for e in c})
    M = sympy.Matrix([[(1 if j < len(reactants) else -1) * counts[j].get(e, 0) for j in range(len(species))] for e in elements])
    ns = M.nullspace()
    if len(ns) != 1:
        raise ValueError("this reaction can't be balanced uniquely (check the species)" if ns else "these species can't form a balanced reaction (atoms don't match)")
    v = ns[0]
    lcm = sympy.ilcm(*[x.q for x in v])
    v = [int(x * lcm) for x in v]
    if all(x < 0 for x in v):
        v = [-x for x in v]
    if any(x <= 0 for x in v):
        raise ValueError("no balance with all-positive coefficients: check reactants vs products")
    g = math.gcd(*v)
    return [x // g for x in v]


# ---------------------------------------------------------------------------
# xTB
# ---------------------------------------------------------------------------

def _calc(numbers, pos_bohr, uhf=0, charge=0, electronic_temperature=None):
    from tblite.interface import Calculator
    c = Calculator("GFN2-xTB", numbers, pos_bohr, charge=charge, uhf=uhf)
    c.set("verbosity", 0)
    c.set("max-iter", 400)
    if electronic_temperature:
        # Fermi smearing: standard for reactive dynamics, keeps the SCF stable while bonds are half-broken
        c.set("temperature", electronic_temperature * KB)
    return c


def _uhf(smiles: str) -> int:
    from rdkit import Chem
    if smiles in TRIPLET:
        return 2
    m = Chem.MolFromSmiles(smiles)
    return sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())


def build(sp: Species) -> Species:
    """RDKit 3D embedding, then GFN2-xTB optimisation."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from scipy.optimize import minimize
    m = Chem.AddHs(Chem.MolFromSmiles(sp.smiles))
    if m.GetNumAtoms() == 1:
        pos = np.zeros((1, 3))
    else:
        AllChem.EmbedMolecule(m, randomSeed=7)
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            AllChem.UFFOptimizeMolecule(m)
        pos = m.GetConformer().GetPositions()
    nums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    uhf = _uhf(sp.smiles)
    charge = Chem.GetFormalCharge(m)
    calc = _calc(nums, pos / BOHR, uhf, charge)

    def f(x):
        calc.update(x.reshape(-1, 3))
        r = calc.singlepoint()
        return float(r.get("energy")), np.asarray(r.get("gradient")).ravel()
    if len(nums) > 1:
        res = minimize(f, (pos / BOHR).ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 300, "gtol": 1e-4})
        x = res.x.reshape(-1, 3)
        e = float(res.fun)
    else:
        x = pos / BOHR
        e = f(x.ravel())[0]
    sp.mol, sp.numbers, sp.positions, sp.energy = m, nums, x * BOHR, e
    return sp


def _bonds(mol) -> set[tuple[int, int]]:
    return {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}


# ---------------------------------------------------------------------------
# complexes and atom mapping
# ---------------------------------------------------------------------------

def _complex(species: list[Species], coeffs: list[int], spacing: float = 3.6):
    """Place molecule copies on a compact grid. Returns numbers, positions (A), bonds, molecule index per atom."""
    copies = [(s, k) for s, c in zip(species, coeffs) for k in range(c)]
    n = len(copies)
    side = math.ceil(n ** (1 / 3)) if n > 4 else max(1, math.ceil(math.sqrt(n)))
    nums, pos, bonds, mol_of = [], [], set(), []
    for idx, (s, _) in enumerate(copies):
        g = np.array([idx % side, (idx // side) % side, idx // (side * side)], float)
        off = (g - (side - 1) / 2) * spacing * (1 if n > 1 else 0)
        p = s.positions - s.positions.mean(axis=0)
        base = len(nums)
        nums += list(s.numbers)
        pos += list(p + off)
        bonds |= {(base + i, base + j) for i, j in _bonds(s.mol)}
        mol_of += [idx] * len(s.numbers)
    return np.array(nums), np.array(pos), bonds, np.array(mol_of)


def _map_atoms(rn, rp, pn, pp, p_mol):
    """Map product atoms onto reactant atoms (same element), moving each product molecule rigidly
    to where its atoms come from, then re-matching (a few rounds of Hungarian assignment)."""
    from scipy.optimize import linear_sum_assignment
    pp = pp.copy()
    perm = np.arange(len(rn))
    for _ in range(6):
        perm = np.empty(len(rn), int)                 # perm[product atom] = reactant atom
        for el in set(rn):
            ri = np.where(rn == el)[0]
            pi = np.where(pn == el)[0]
            cost = ((pp[pi][:, None, :] - rp[ri][None, :, :]) ** 2).sum(-1)
            a, b = linear_sum_assignment(cost)
            perm[pi[a]] = ri[b]
        for m in set(p_mol):
            idx = np.where(p_mol == m)[0]
            pp[idx] += rp[perm[idx]].mean(axis=0) - pp[idx].mean(axis=0)
    return perm, pp


# ---------------------------------------------------------------------------
# steered reaction dynamics
# ---------------------------------------------------------------------------

def simulate_reaction(reactants, products, coeffs, steps=1600, dt_fs=0.5, temperature=600.0,
                      record_every=8, progress=None) -> dict:
    nr = len(reactants)
    rn, rp, rbonds, r_mol = _complex(reactants, coeffs[:nr])
    pn, pp, pbonds_p, p_mol = _complex(products, coeffs[nr:])
    if sorted(rn) != sorted(pn):
        raise ValueError("atom counts differ between sides")
    perm, pp_aligned = _map_atoms(rn, rp, pn, pp, p_mol)
    # product bonds expressed in reactant atom indices
    pbonds = {(min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in pbonds_p}
    form = pbonds - rbonds
    brk = rbonds - pbonds
    target = {}
    for i, j in pbonds:
        pi_ = int(np.where(perm == i)[0][0]); pj_ = int(np.where(perm == j)[0][0])
        target[(i, j)] = float(np.linalg.norm(pp[pi_] - pp[pj_]))
    nums = rn
    x = rp / BOHR
    masses = np.array([_mass(z) for z in nums])[:, None] * AMU
    rng = np.random.default_rng(0)
    kT = KB * temperature
    v = rng.normal(0, 1, x.shape) * np.sqrt(kT / masses)
    v -= (v * masses).sum(0) / masses.sum()
    dt = dt_fs * FS
    gamma = 0.02 / FS                       # Langevin friction (1/50 fs)
    uhf = 0 if sum(nums) % 2 == 0 else 1
    calc = _calc(nums, x, uhf, electronic_temperature=3000.0)
    k_max = 0.06                            # hartree / bohr^2 bias at full strength
    frames, energies, bias_e, bond_frames, times = [], [], [], [], []
    events = []
    prev_bo = None
    t0 = time.time()

    last = {"res": None}

    def forces(x, step):
        calc.update(x)
        try:          # restart from the previous wavefunction: faster and far more robust mid-reaction
            res = calc.singlepoint(last["res"], copy=True) if last["res"] is not None else calc.singlepoint()
        except Exception:
            calc.set("temperature", 6000.0 * KB)
            res = calc.singlepoint()
            calc.set("temperature", 3000.0 * KB)
        last["res"] = res
        e = float(res.get("energy"))
        g = np.asarray(res.get("gradient")).copy()
        ramp = min(1.0, step / (0.6 * steps))
        k = k_max * ramp
        eb = 0.0
        for (i, j) in form:
            d = x[i] - x[j]; r = np.linalg.norm(d); r0 = target[(i, j)] / BOHR
            if r > r0:
                eb += 0.5 * k * (r - r0) ** 2
                gr = k * (r - r0) * d / r
                g[i] += gr; g[j] -= gr
        for (i, j) in brk:
            d = x[i] - x[j]; r = np.linalg.norm(d); r1 = 2.8 / BOHR
            if r < r1:
                eb += 0.5 * k * (r1 - r) ** 2
                gr = -k * (r1 - r) * d / r
                g[i] += gr; g[j] -= gr
        return e, eb, g, res

    e, eb, g, res = forces(x, 0)
    for step in range(steps + 1):
        if step % record_every == 0:
            bo = _bond_orders(res)
            frames.append((x * BOHR).round(4).tolist())
            energies.append(e)
            bias_e.append(eb)
            times.append(step * dt_fs)
            bl = [[int(i), int(j), round(float(bo[i, j]), 2)] for i in range(len(nums)) for j in range(i + 1, len(nums)) if bo[i, j] > 0.3]
            bond_frames.append(bl)
            if prev_bo is None:
                stable = bo > 0.5
                pending = np.zeros_like(stable, dtype=int)
                state = stable.copy()
            else:
                # hysteresis: a bond forms above 0.65 and breaks below 0.35 (ionic/partial bonds sit near 0.5)
                state = np.where(stable, bo > 0.35, bo > 0.65)
                changed = state != stable
                pending = np.where(changed, pending + 1, 0)
                for i, j in zip(*np.where(np.triu(pending >= 3, 1))):
                    stable[i, j] = stable[j, i] = state[i, j]
                    pending[i, j] = pending[j, i] = 0
                    events.append({"t_fs": (step - 2 * record_every) * dt_fs, "type": "formed" if state[i, j] else "broke",
                                   "atoms": [int(i), int(j)], "pair": f"{_sym(nums[i])}–{_sym(nums[j])}"})
            prev_bo = bo
            if progress:
                progress(step / steps)
        if step == steps:
            break
        # BAOAB Langevin step
        v -= 0.5 * dt * g / masses
        x = x + 0.5 * dt * v
        c1 = math.exp(-gamma * dt)
        v = c1 * v + np.sqrt((1 - c1 * c1) * kT / masses) * rng.normal(0, 1, x.shape)
        x = x + 0.5 * dt * v
        e, eb, g, res = forces(x, step + 1)
        v -= 0.5 * dt * g / masses
    # did the expected products form?
    bo = _bond_orders(res)
    got = sum(1 for (i, j) in pbonds if bo[i, j] > 0.5)
    stray = sum(1 for (i, j) in rbonds - pbonds if bo[i, j] > 0.5)
    e_rel = [(ee - energies[0]) * HARTREE_KJ for ee in energies]
    return {
        "numbers": nums.tolist(), "symbols": [_sym(z) for z in nums], "frames": frames, "bonds": bond_frames,
        "times_fs": times, "energy_kj": [round(v_, 2) for v_ in e_rel],
        "bias_kj": [round(b * HARTREE_KJ, 2) for b in bias_e], "events": events,
        "product_bonds_formed": got, "product_bonds_expected": len(pbonds), "reactant_bonds_remaining": stray,
        "temperature_K": temperature, "dt_fs": dt_fs, "steps": steps, "seconds": round(time.time() - t0, 1),
        "peak_energy_kj": round(max(e_rel), 1),
    }


_PT = None
def _bond_orders(res) -> np.ndarray:
    """Wiberg bond orders. tblite 0.4.0 left debug print() calls in this code path; swallow them."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return np.asarray(res.get("bond-orders"))


def _mass(z):
    global _PT
    from rdkit import Chem
    _PT = _PT or Chem.GetPeriodicTable()
    return _PT.GetAtomicWeight(int(z))


def _sym(z):
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


# ---------------------------------------------------------------------------
# full pipeline + background jobs
# ---------------------------------------------------------------------------

def analyse(text: str) -> dict:
    reactants, products = parse(text)
    coeffs = balance(reactants, products)
    for s in reactants + products:
        build(s)
    nr = len(reactants)
    de_xtb = (sum(c * s.energy for c, s in zip(coeffs[nr:], products)) -
              sum(c * s.energy for c, s in zip(coeffs[:nr], reactants))) * HARTREE_KJ
    allsp = reactants + products
    nu = [-c for c in coeffs[:nr]] + list(coeffs[nr:])                 # stoichiometric numbers: products +, reactants -
    dh = sum(v * s.dhf for v, s in zip(nu, allsp)) if all(s.dhf is not None for s in allsp) else None
    dh_gas = sum(v * s.dhf_gas for v, s in zip(nu, allsp)) if all(s.dhf_gas is not None for s in allsp) else None
    # the balance, element by element (exact integers from the null space; re-checked here)
    elems: dict[str, list[int]] = {}
    for v, s in zip(nu, allsp):
        for el, n in _formula_counts(s.smiles).items():
            e = elems.setdefault(el, [0, 0])
            e[0 if v < 0 else 1] += abs(v) * n
    def fmt(side, cs):
        return " + ".join((f"{c} " if c > 1 else "") + _pretty(s) + (f"({s.phase})" if s.phase != "g" else "(g)") for s, c in zip(side, cs))
    n_atoms = sum(c * len(s.numbers) for c, s in zip(coeffs, allsp)) // 2
    first = reactants[0]
    return {
        "equation": f"{fmt(reactants, coeffs[:nr])} → {fmt(products, coeffs[nr:])}",
        "coefficients": coeffs,
        "balance": [{"element": el, "left": a, "right": b} for el, (a, b) in sorted(elems.items())],
        "balanced": all(a == b for a, b in elems.values()),
        "species": [{"name": s.name, "smiles": s.smiles, "formula": _pretty(s), "side": "reactant" if k < nr else "product",
                     "coefficient": coeffs[k], "symbols": [_sym(z) for z in s.numbers], "positions": s.positions.round(4).tolist(),
                     "bonds": [[int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), float(b.GetBondTypeAsDouble())] for b in s.mol.GetBonds()],
                     "xtb_energy_hartree": round(s.energy, 6), "dHf_kJmol": s.dhf, "phase": s.phase, "dHf_gas_kJmol": s.dhf_gas}
                    for k, s in enumerate(allsp)],
        "dH_std_kJmol": round(dh, 2) if dh is not None else None,
        "dH_gas_kJmol": round(dh_gas, 2) if dh_gas is not None else None,
        "per_mol_first": {"formula": _pretty(first), "kJmol": round(dh / coeffs[0], 2)} if dh is not None else None,
        "thermo_source": "measured standard enthalpies of formation, 298.15 K, 1 bar (NIST Chemistry WebBook / CRC Handbook)",
        "dE_xtb_kJmol": round(de_xtb, 1),
        "xtb_note": "GFN2-xTB is a fast semi-empirical quantum method built for structures and dynamics; its reaction energies can be off "
                    "by hundreds of kJ/mol (even the sign), so it is shown only as a rough electronic-energy estimate.",
        "dH_exp_kJmol": round(dh, 1) if dh is not None else None,              # kept for older clients
        "exothermic": (dh < 0) if dh is not None else (dh_gas < 0 if dh_gas is not None else None),
        "atoms": n_atoms,
        "_objects": (reactants, products, coeffs),
    }


def _pretty(s: Species) -> str:
    if s.smiles in DISPLAY:
        return DISPLAY[s.smiles]
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import Chem
    return CalcMolFormula(Chem.MolFromSmiles(s.smiles))


JOBS: dict[str, dict] = {}


def start_job(text: str) -> dict:
    info = analyse(text)
    reactants, products, coeffs = info.pop("_objects")
    if info["atoms"] > 60:
        raise ValueError(f"{info['atoms']} atoms is too many for live quantum dynamics on this machine (limit 60)")
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "state": "running", "progress": 0.0, "info": info, "started": time.time()}
    steps = 1200 if info["atoms"] <= 20 else 1600

    def run():
        try:
            traj = simulate_reaction(reactants, products, coeffs, steps=steps,
                                     progress=lambda p: JOBS[jid].__setitem__("progress", round(p, 3)))
            JOBS[jid].update(state="done", trajectory=traj)
        except Exception as e:
            log.exception("reaction failed")
            JOBS[jid].update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()
    return JOBS[jid]
