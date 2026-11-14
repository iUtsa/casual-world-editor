"""Chemistry: balanced equations and reaction enthalpies against textbook standard values (25 °C, 1 bar)."""
import pytest

from backend.app import chem

# reaction -> (textbook ΔH° of the balanced equation as written, kJ/mol)
CASES = {
    "methane + oxygen -> carbon dioxide + water": -890.6,            # ΔcH°(CH4) = -890.3 ... -890.8
    "propane + oxygen -> carbon dioxide + water": -2219.9,
    "ethanol + oxygen -> carbon dioxide + water": -1366.8,
    "hydrogen + oxygen -> water": -571.7,                             # 2 × -285.83
    "nitrogen + hydrogen -> ammonia": -91.8,                          # Haber process
    "sulfur dioxide + oxygen -> sulfur trioxide": -197.8,             # contact process
    "hydrogen peroxide -> water + oxygen": -196.1,                    # 2 H2O2(l) -> 2 H2O(l) + O2
    "hydrogen + chlorine -> hydrogen chloride": -184.6,
}


@pytest.mark.parametrize("text,ref", list(CASES.items()))
def test_reaction_enthalpy_matches_textbook(text, ref):
    a = chem.analyse(text)
    assert a["balanced"]
    assert a["dH_std_kJmol"] == pytest.approx(ref, abs=1.0), a["equation"]
    assert a["exothermic"] is (ref < 0)


def test_gas_phase_differs_by_measured_vaporisation():
    a = chem.analyse("methane + oxygen -> carbon dioxide + water")
    assert a["dH_gas_kJmol"] == pytest.approx(-802.6, abs=1.0)          # water as vapour: 2 × 44.0 kJ/mol less
    assert a["dH_std_kJmol"] - a["dH_gas_kJmol"] == pytest.approx(-2 * 44.0, abs=0.2)


BENCH = ["methane + oxygen -> carbon dioxide + water", "hydrogen + oxygen -> water", "nitrogen + hydrogen -> ammonia",
         "ethene + hydrogen -> ethane", "carbon monoxide + oxygen -> carbon dioxide", "hydrogen + chlorine -> hydrogen chloride",
         "hydrogen peroxide -> water + oxygen", "sulfur dioxide + oxygen -> sulfur trioxide", "ethanol + oxygen -> carbon dioxide + water",
         "propane + oxygen -> carbon dioxide + water"]


def test_dft_benchmark():
    """The quantum-chemistry heat of reaction (used when a molecule has no measured data) against measured gas-phase values.
    The app quotes thermo_dft.BENCH_MAE_KJ as its uncertainty, so it must stay true."""
    import numpy as np
    from backend.app import thermo_dft as T
    errs = []
    for text in BENCH:
        rs, ps = chem.parse(text)
        cs = chem.balance(rs, ps)
        nu = [-c for c in cs[:len(rs)]] + list(cs[len(rs):])
        for s in rs + ps:
            chem.build(s)
        dh = sum(v * T.enthalpy(s)["H"] for v, s in zip(nu, rs + ps)) * T.HARTREE_KJ
        errs.append(dh - sum(v * s.dhf_gas for v, s in zip(nu, rs + ps)))
    mae = float(np.mean(np.abs(errs)))
    assert mae <= T.BENCH_MAE_KJ + 1.0, f"MAE {mae:.1f} kJ/mol"
    assert max(abs(e) for e in errs) < 80
