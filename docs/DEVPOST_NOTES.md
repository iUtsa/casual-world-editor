# Causal World Editor: Devpost, slides and video notes

A running record of what we built, what we measured and what went wrong, kept as we work so the Devpost write-up,
the slides and the demo video can be built from it. Numbers are measured on our M1 Mac (8 GB) unless noted.
Newest entries in the build log go at the bottom.

## Inspiration

Students go blank at physics diagrams: the picture is still, the physics is not, and they have to "play the movie
in their head" (cognitive scientists call this mental animation; Mary Hegarty's research on mechanical diagrams).
Labs that let you try things cost equipment most schools don't have. We wanted the diagram, or the student's own
desk, to move, and to answer "what if?" with numbers you can check.

**Tagline:** Stop imagining. Start simulating.

## What it does

1. **See:** point a laptop or phone camera at a desk, or snap a photo of a textbook diagram.
2. **Build:** in seconds, a 3D physics world: every object with size, mass, friction and what it rests on.
3. **Ask "what if?":** type or say it ("drop a baseball on the book", "what if gravity were the Moon's?"), or use
   your real hand: gestures (pinch to grab, flick to throw, swipe to push), or **🖐 Touch**, where your hand IS
   physics: 21 finger bones push, tip and flick objects live, and the session replays exactly afterwards.
4. **Watch:** a real physics engine simulates it; the result plays over the live camera or in an orbitable 3D twin.
5. **Check:** measured numbers (speed, force, energy) beside the textbook prediction, with % agreement.

Studio extends the same "what if" to space (real NASA positions), circuits, chemistry (measured and quantum
reaction energies) and an AI Lab that shows GPT-2 thinking.

## How we built it

| Part | Technology | Notes |
| --- | --- | --- |
| Depth | Depth Anything V2 (metric, small) | on-device, ~0.5 s per frame on the M1 GPU |
| Object outlines | SAM 2.1 tiny, prompted from geometry | dense "segment everything" was too slow on 8 GB |
| 3D world | our table finding, shape fitting, support graph | boxes, cylinders, revolved profiles |
| Multi-angle 3D scan | our structure-from-motion + voxel carving | SIFT, PnP, Schur-complement bundle adjustment, label-consistent carving |
| Physics | MuJoCo, calibrated contacts | restitution calibrated per material by bisection |
| Hands | MediaPipe HandLandmarker in the browser | metric 3D hand by a linear solve |
| Language | Qwen2.5-VL-3B via MLX, locally | names objects, reads diagrams |
| Voice | Whisper base, locally | |
| Space | REBOUND IAS15 + NASA JPL Horizons | tidal disruption, inspiral, galaxy mergers |
| Chemistry | RDKit, GFN2-xTB, PySCF (M06-2X/def2-TZVP) | |
| App | FastAPI backend, React + React Three Fiber frontend | phone camera over local HTTPS |

Everything runs on the laptop, free, offline; camera, voice and diagrams never leave the machine.

## Measured results

| Claim | Number |
| --- | --- |
| Agreement with textbook physics (19 checks) | mean 99.89 %, worst 99.55 % |
| Automated tests | 150+ passing |
| Chemistry, quantum reaction energies vs measured | mean error 21.6 kJ/mol (10 reactions) |
| Hand tracking | ~13 ms per frame (M1 GPU) |
| One-photo desk scan | ~3-4 s |
| Multi-angle scan: camera positions from video alone | median 0.9 cm, worst 2.5 cm (60-frame synthetic sweep) |
| Multi-angle scan: carved object sizes (true poses) | within ~1 cm of ground truth |
| Live hand physics | 2.2 ms per update (budget 33 ms); replay matches live exactly (0.00 mm) |
| Table plane from multi-view geometry | 0.00 deg tilt error, 2.9 mm rms |
| Objects found by plane + parallax (no depth model) | 5 of 6 on the photoreal test desk (one-photo method: 1 of 6) |
| Multi-angle scan, end to end from video (photoreal desk) | lemon 7.1 mm, box 9.3 mm (volume 1458 vs 1477 cm³), can 9.8 mm, bowl 10.5 mm surface error; alarm clock failed |
| Multi-angle scan time (56 frames, M1 8 GB) | 308 s (was 707 s before vectorising and faster mask upsampling) |

## Challenges we ran into (with what fixed them)

- **8 GB of memory:** SAM 2's video tracker swapped for over 10 minutes; replaced it with geometry-guided prompts
  (constant memory). PySCF next to PyTorch deadlocked (two OpenMP runtimes); moved quantum chemistry to its own
  process.
- **Monocular depth lies on synthetic scenes:** Depth Anything was 2-3x off on rendered test scenes, so the table
  wasn't even flat. We stopped trusting it for geometry: the table plane and object discovery now come from
  multi-view geometry, and depth only sets the overall scale.
- **Camera solving from plain video (iPhone Safari gives no tracking):** bundle adjustment took 130 s and lost
  frames; a hand-written Levenberg-Marquardt with the Schur complement brought it to ~36 s, and re-placing badly
  started cameras (instead of dropping their points) took worst-case error from 48 cm to 2.5 cm.
- **Visual hull "ghosts":** carving left blobs between objects and roofs on flat tops (book came out 16 cm thick
  instead of 3.5). Fixed with label-consistent space carving, a limit to each object's observed surface, and
  free-space votes from depth.
- **MuJoCo's own camera:** our ground-truth camera was off by 30 px because MuJoCo's camera[0] is the left stereo
  eye.
- **Hands in physics:** kinematic bones whose poses are interpolated every 2 ms step, so a fast flick never
  teleports into an object; mocap bodies keep saved states compatible.
- **🖐 Touch end-to-end bugs found by a scripted browser test:** the server didn't answer an idle message (the page
  keeps one in flight, so it froze); a simulation with zero segments crashed the timeline; fast swipes pushed
  objects with no contact ever reported (contacts lasted milliseconds between replies; now the peak contact of every
  physics step is reported); sessions only ended after a reported touch (now also after anything moved).
- **Live vs replay drift (27 mm!):** a live hand session and its replay were each deterministic but ended 27 mm
  apart. Cause: the recording was rounded to micrometres and contact physics amplified it. Using the rounded values
  live made the replay match to 0.00 mm. A second bug: simulating on network arrival times squeezed a burst of
  hand poses into milliseconds; we now simulate on the camera's timestamps.

## Accomplishments we're proud of

- Every answer checked against textbook physics, 99.9 % average agreement.
- A full multi-angle 3D reconstruction pipeline written from scratch for an 8 GB laptop.
- Runs entirely offline and free, so any school can use it.

## What we learned

- Never let a model's guess shape physics: measure, then show the measurement beside the theory.
- On small hardware, geometry beats bigger models: plane + parallax found objects the depth model couldn't.
- Test against ground truth: every bug above was found by a benchmark, not by looking at the demo.

## What's next

- Relocalize the camera against the 3D scan so it can move after scanning.
- Classroom pilots; teacher-made "what if" lesson packs.

## Demo assets and video shot list

- Printed diagrams that read cleanly: (to fill after testing)
- Desk setup: 3-5 clear objects, phone on a stand at 30-60 deg, even light.
- Shots: blank-stare opening; snap a diagram -> it moves -> "what if 60 deg?"; desk scan -> outlines; "drop a
  baseball on the book"; hand flick; 3D twin orbit; space what-if; theory-check panel; closing line.
- Backup: full screen recording of the demo.

## Build log

- **2026-11-11 to 11-13:** core app: perception, MuJoCo physics, Live AR, Studio labs (physics, space, circuits,
  chemistry, AI Lab), phone camera, voice, diagram reading, hand gestures, theory checks.
- **2026-11-13:** multi-angle scan research: carving validated (~1 cm with true poses); camera solve from video.
- **2026-11-14:** geometry-guided masks (mean IoU 0.90); sweep scan pipeline, API, phone recording, progress UI;
  scanned meshes in physics and 3D views; photoreal test desk; plane + parallax discovery (5/6 objects); live
  hand physics core (2.2 ms/update, 3 mm replay match); code pushed to GitHub file by file.
- **2026-11-14 (later):** sweep scan now finds objects by multi-view parallax (exact table plane, 3D-support filter
  drops false detections, half the mask work); live hand WebSocket /ws/hand (client timestamps, commits a replayable
  "hand" action; replay identical to live, end-to-end test passing).
- **2026-11-14 (later):** 🖐 Touch in the browser: metric 3D hand (MediaPipe world landmarks + linear solve), One Euro
  smoothing, WebSocket streaming with camera timestamps (one message in flight), live poses in every 3D view, glowing
  contact markers with force in newtons, session ends 1.5 s after the hand leaves and the replay plays. Scan speed:
  silhouette prediction vectorised (20k points in 7 ms), SAM mask upsampling done with OpenCV (identical masks).
- **2026-11-14 (evening):** 🖐 Touch verified end to end in Chromium (scripted 3D hand -> live contact markers ->
  session ends -> replay recorded as a "hand" action -> chat report; no page errors). Toolbar now overlaps the hint
  pill (7 buttons): fix in the UI redesign.
- **2026-11-14 (night):** landing page + one design system. "/" is now the landing page (paper and ink, offset
  shadows, Bricolage Grotesque / Instrument Sans / IBM Plex Mono); every "Get started" opens the lab at "/lab/" with
  an ink page-wipe. Landing additions: scroll reveals, count-up stats, reading-progress bar, active-section nav, a
  live 🖐 Touch toy (2D rigid bodies, the pointer is a kinematic hand, contact forces in newtons), an animated
  walk-around-scan explainer, a fourth "Use your hand" way, current numbers (151 tests) and honest limits. The lab
  was restyled to the same theme (header, chat, cards, Studio, chemistry, AI Lab, modals, camera overlays), and the
  Live AR tools moved off the camera into a grouped dock (Scan · Play · Result), so nothing covers the picture.
  Verified with screenshots at desktop and phone widths (no horizontal overflow, no page errors), the Touch e2e on
  /lab/, and 151 backend tests.

## Demo notes from the redesign

- Open on the landing page: the hero desk demo answers the chips, the Touch toy is a 10-second "your hand is
  physics" moment before switching to the real lab.
- Watch out: with the scripted e2e hand the peak contact marker read about 1500 N (a fast sweep starting inside an
  object). Real hands come from above and read lower, but check the numbers before recording the video.
