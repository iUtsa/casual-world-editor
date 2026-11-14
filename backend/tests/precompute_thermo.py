"""Fill data/thermo_cache.json with the quantum thermochemistry of every molecule in chem.SPECIES (run once; minutes)."""
import sys, time
sys.path.insert(0, ".")
from backend.app import chem, thermo_dft as T
done = set()
for key, (smi, *_rest) in chem.SPECIES.items():
    if smi in done:
        continue
    done.add(smi)
    t = time.time()
    try:
        sp = chem.resolve(key); chem.build(sp); h = T.enthalpy(sp)
        print(f"{sp.name:24} {time.time() - t:6.1f} s  imag={h['n_imag']}", flush=True)
    except Exception as e:
        print(f"{key:24} FAILED {e}", flush=True)
