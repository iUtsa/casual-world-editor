# Causal World Editor

**Stop imagining. Start simulating.**

Students stare at still diagrams in textbooks (a block on a ramp, two balls about to collide, a planet's orbit) and
try to play the movie in their heads. Cognitive scientists call this *mental animation*, and it is hard. Causal World
Editor plays the movie for them, then lets them ask **"what if?"**

Point a camera at your desk, or snap a photo of a diagram. The app builds a 3D physics world from what it sees. Then you
can push things with your hand, drop objects that aren't there, change gravity, or ask a question in plain English.
Every result is simulated, measured, and **checked against textbook theory**, so you get an answer you can verify, not a
guess.

Everything runs **free and on-device** (built and tested on an M1 MacBook with 8 GB). Camera frames, voice and diagrams
never leave the machine.

---

## What you can do

### 📷 Live AR: your real desk becomes a physics world
- **Scan** the desk with the laptop webcam or your **phone** (scan with the phone, and the result appears on the laptop).
- Every object is lifted into 3D with size, mass, friction and what it rests on. A **support graph** shows which object
  holds up which.
- **Switch to 3D** to orbit a photo-textured digital twin of your desk.
- **Click any object** to inspect it: size, mass, friction, and after it moves its top speed, distance travelled,
  hardest hit and energy, plus a speed chart.
- **✋ Hand control** (MediaPipe, in the browser): pinch to grab and throw, or swipe your palm to push. The hand moves the
  object's **digital twin**, not the real one, so you can knock the mug over without spilling your coffee.
- **Say or type what you want**, and it acts on the scanned scene, not an imaginary one:
  *"knock over the bottle"*, *"throw a baseball at the mug"*, *"drop a baseball on the book"*,
  *"what if the book was 10× heavier?"*, *"make the table icy"*, *"moon gravity"*.
  Directions are camera-relative, throws are solved ballistically, and each action continues from the settled result of
  the previous one.
- **📷 Snap a diagram**: photograph a textbook or whiteboard diagram. A local vision model reads it, shows you an
  editable reading card (*"inclined plane, 30°, 2 kg block…"*), and then **▶ Simulate this** runs it. Follow up with
  *"what if the angle was 60°?"*

### 🧪 Studio: every lab in one place
| Tab | Ask things like | Under the hood |
|---|---|---|
| **⚛ Physics** | *"two blue whales collide at 10 m/s"*, *"a toy car pulls a truck"*, *"line up 10 dominoes and knock them over"* | MuJoCo, with real-world sizes and masses, toy scaling (mass ∝ k³, speed ∝ √k) and measurements from simulator state |
| **🪐 Space** (in Physics) | *"what if Jupiter orbited Earth?"*, *"what if there was no Sun?"*, *"the Sun falls into a black hole"*, *"two neutron stars merge"*, *"Milky Way and Andromeda collide"* | REBOUND IAS15 with real NASA JPL Horizons positions; tidal disruption, gravitational-wave inspiral, merger and galaxy-collision physics |
| **🔌 Circuits** (in Physics) | *"9 V battery with two bulbs in parallel"*, *"RC circuit with 100 µF and 1 kΩ"* | Modified nodal analysis (the method SPICE uses), with transients |
| **🧪 Chemistry** | *"burn methane"*, *"hydrogen and oxygen make water"* | Exact balancing, measured NIST/CRC enthalpies, GFN2-xTB molecular dynamics, and **quantum chemistry (DFT)** for molecules with no measured data |
| **🧠 AI Lab** | *"the cat sat on the"* | A real GPT-2 running locally: attention heads, logit lens, and the answer forming layer by layer |

### ✅ Every answer is checked
Results come with a **theory check**: the measured value next to the textbook prediction and the agreement between them.
Examples are conservation of energy and momentum, ballistic range, Kepler's laws, tidal radius, debris fall-back time,
gravitational-wave chirp, and Kirchhoff's laws. When a textbook formula doesn't apply (for example a tidal disruption with
a small mass ratio), the app says so instead of reporting a misleading score.

---

## Quick start

Requirements: macOS on Apple Silicon, Python 3.12, Node 18+.

```bash
# 1. Main app: Python environment, ~250 MB of vision models, UI build, synthetic test scenes
./setup.sh
.venv/bin/pip install rebound reboundx pyscf rdkit tblite sympy scipy sentence-transformers qrcode trimesh pillow

# 2. Local vision-language model worker (Qwen2.5-VL-3B, 4-bit, runs on the Apple GPU via MLX)
python3.12 -m venv .venv-ai
.venv-ai/bin/pip install mlx-vlm

# 3. Run
./restart.sh     # stops any old instance, starts the server, waits until the models are ready
```

- **Laptop:** open <http://localhost:8000>
- **Phone as a camera:** click **Connect phone** and scan the QR code. The phone opens `https://<your LAN IP>:8443/phone`,
  served over HTTPS with a certificate generated on first run, because phone browsers only allow the camera on HTTPS.
  Frames travel over your local network only.
- **Development with hot reload:** run `./dev.sh`, which serves the backend on :8000 and the Vite UI on :5173.

### Optional cloud features (off by default)
The app is fully functional without any keys. If you set them:
- `ANTHROPIC_API_KEY` uses Claude for more fluent explanations. A deterministic writer produces the same explanation
  structure offline.
- `FAL_KEY` lets you generate a 3D model for objects that aren't in the library (only the text prompt is sent).

---

## How it works

```
camera / phone / diagram photo
        │
        ▼
 perception ──────────► world model ──────────► simulation ──────────► measurement + theory check ──► UI
 depth (Depth Anything V2)  objects, sizes, masses,   MuJoCo (desk)              60 Hz state: speed, forces,       Live AR overlay,
 masks (SAM 2.1)            friction, support graph,  REBOUND (space)            energy; textbook formulas side    3D twin, charts,
 table plane, 3D lifting    interventions + undo      MNA (circuits)             by side with measured values      explanation
 local VLM (labels,                                   xTB / DFT (chemistry)
 diagrams), Whisper (voice)
```

| Part | Implementation |
|---|---|
| Depth | Depth Anything V2 metric-indoor-small; multi-frame median fusion from the phone; or sensor depth (RGB-D) |
| Table and segmentation | RANSAC plane fit (the highest of several parallel planes is the table), SAM 2.1-tiny prompted from geometry, convex/concave seam test to separate stacked objects |
| 3D lifting | Footprint in the table frame, box vs. cylinder from surface curvature, revolved meshes for round objects, hidden sides mirrored from the visible ones |
| Physics | MuJoCo with elliptic friction cones; checkpoints so each action continues from the last settled state (0.4–1.3 s per action instead of replaying the whole history) |
| Language → actions | `scene_cmd.py` for direct commands on the scanned scene (deterministic: camera-relative directions, ballistic throw solving, push speed from friction and distance); `experiment.py` for general experiments |
| Physical knowledge | Scanned object → 3D library model → table of real sizes and masses → on-device model estimate, accepted only if the implied density is physically plausible |
| Space | REBOUND IAS15 (energy conserved to ~1e-15), JPL Horizons ephemerides cached for offline use, tidal disruption (frozen-in model), Peters inspiral, numerical-relativity fits for merger energy and spin |
| Chemistry | Measured ΔHf° where it exists; otherwise M06-2X/def2-TZVP (PySCF) on GFN2-xTB geometries with xTB vibrations, run in a separate process |
| Hands | MediaPipe HandLandmarker in the browser (~13 ms per frame on M1 GPU), pinch with hysteresis, palm velocity for pushes |
| Rendering | React Three Fiber; moved objects are projectively textured with their real pixels, with real objects as occluders |
| Local AI | Qwen2.5-VL-3B (MLX, 4-bit) for labels and diagram reading; Whisper base.en for voice; both run in their own process on 127.0.0.1 |

---

## Accuracy and tests

```bash
.venv/bin/python -m pytest -q backend/tests           # physics, perception, space, chemistry, hands
.venv/bin/python backend/tests/accuracy_report.py     # theory-check agreement across a battery of experiments
.venv/bin/python backend/tests/e2e/ui_flow.py <out>   # browser end-to-end (Playwright); see backend/tests/e2e/
```

As of 13 Nov 2026:
- **145 tests passed**, plus 1 expected failure (automatic segmentation can under-split a tower of similar boxes).
- **Theory-check agreement:** mean 99.89%, worst 99.55% across the experiment battery.
- **Chemistry (DFT):** mean absolute error of 21.6 kJ/mol on 10 reactions with measured gas-phase enthalpies. The app
  reports ± max(21.6 kJ/mol, 4.5% of |ΔH|).
- **Speed on M1 with 8 GB:** a scan takes ~3–4 s warm, and a desk simulation takes 30–90 ms.

The tests check physics, not just code paths. For example, a resting contact force must equal the object's weight, and
energy given to a passive object must equal the energy the other object gave up.

---

## Known limitations

- **One camera frame per scan.** Hidden sides are inferred, handles aren't modelled, and shapes are boxes, cylinders or
  revolved profiles.
- **Monocular scale can be off.** Use **Calibrate scale** with one object's real height, or RGB-D input.
- **The camera must stay fixed after a scan.** There's no continuous tracking. The phone detects movement and asks for a
  rescan.
- **The small local vision model can misread** messy handwriting or unusual objects. That's why the diagram reading is
  shown for editing before anything runs.
- **Memory:** on 8 GB, close other heavy apps before a demo.
- **Chemistry DFT** runs about 3–4% low on large combustion reactions; the reported uncertainty covers this.
- **Tidal disruptions** with a mass ratio below 100 are simulated but not scored, because the textbook formulas don't
  apply there.
- **Hand control** has been verified with scripted landmarks and real-time detection; rehearse with a real hand in your
  lighting.

---

## Project layout

```
backend/
  app/
    main.py            FastAPI app and all /api routes
    perception/        depth, segmentation, lifting, labels, materials
    physics/sim.py     MuJoCo scene building, interventions, checkpoints
    scene_cmd.py       direct commands on the scanned scene
    experiment.py      general experiments from plain language
    lab.py             Studio runs and "what if" follow-ups
    orbits.py, encounters.py, galaxy.py, ephemeris.py   space
    circuits.py        circuit solver
    chem.py, thermo_dft.py                              chemistry
    transformer_lab.py AI Lab (GPT-2)
    diagram.py         reading diagram photos
    measure.py, exp_analysis.py   measurements and theory checks
    local_ai.py, speech.py        on-device AI clients
    remote.py, live.py, static/phone.html   phone camera
  ai_worker.py         local vision-language model worker (.venv-ai)
  tests/               unit tests, accuracy report, e2e browser tests
frontend/src/
  App.tsx              Live AR | Studio shell, chat, hand actions
  components/          LiveAR, Twin3D, HandLayer, ObjectCard, SpaceView, ChemView, ...
  hands.ts             MediaPipe hand tracking
assets/                CC0 3D object library (licence per asset in LICENSE.json)
```

## Credits

- 3D models: [Poly Haven](https://polyhaven.com) (CC0) and the Khronos glTF sample models. Each asset's source and
  licence are recorded in `assets/library/<id>/LICENSE.json`.
- Models: Depth Anything V2, SAM 2.1, Qwen2.5-VL, Whisper, GPT-2, MediaPipe Hand Landmarker.
- Physics and chemistry: MuJoCo, REBOUND, PySCF, tblite (GFN2-xTB), RDKit.
- Ephemerides: NASA JPL Horizons.
