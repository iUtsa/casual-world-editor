/* Casual World Editor landing: motion and interaction layer. No dependencies.
   - scroll reveal, count-up stats, nav progress + active section, hero tilt, page wipe into the lab
   - 🖐 Touch toy: a small 2D rigid-body sim where the pointer is a kinematic hand
   - walk-around scan animation: phone circles the desk, cameras + points appear, objects are carved */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = matchMedia("(pointer: fine)").matches;

  // ---------------------------------------------------------------- hero words
  $$(".split .w").forEach((w, i) => w.style.setProperty("--i", i));

  // ---------------------------------------------------------------- reveal on scroll (staggered per group)
  const groups = [
    ".section-head", ".two-col > *", ".ways-grid > *", ".flow > *", ".sim-grid > *", ".tabs", ".pipe > .stage-card",
    ".how-more > *", ".stats > *", ".laws", ".honest", ".limits-grid > *", ".steps > *", ".touch-grid > *", ".cta-inner > *",
    ".ways h2", ".demo",
  ];
  groups.forEach((sel) => $$(sel).forEach((el, i) => {
    if (el.closest(".hero")) return;
    el.classList.add("reveal");
    el.style.setProperty("--d", Math.min(i, 6));
  }));
  const countUp = (el) => {
    const to = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0);
    if (!to || reduce) return;
    const t0 = performance.now(), dur = 1200;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = (to * e).toFixed(dec);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in");
    $$("[data-count]", e.target).forEach(countUp);
    io.unobserve(e.target);
  }), { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  $$(".reveal").forEach((el) => io.observe(el));

  // ---------------------------------------------------------------- nav: scroll progress + active section
  const nav = $(".nav"), bar = $(".nav-progress span");
  const onScroll = () => {
    const h = document.documentElement;
    const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
    bar && bar.style.setProperty("--p", p.toFixed(4));
    nav && nav.classList.toggle("scrolled", h.scrollTop > 8);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const links = new Map($$(".nav-links a").map((a) => [a.getAttribute("href").slice(1), a]));
  const secIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    links.forEach((x) => x.removeAttribute("aria-current"));
    links.get(e.target.id)?.setAttribute("aria-current", "true");   // sections with no nav link clear it
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main > section").forEach((s) => secIO.observe(s));

  // ---------------------------------------------------------------- hero panel tilt (mouse only)
  const desk = $("#desk"), hero = $(".hero");
  if (desk && hero && fine && !reduce) {
    hero.addEventListener("pointermove", (e) => {
      const r = desk.getBoundingClientRect();
      const x = (e.clientX - (r.left + r.width / 2)) / r.width, y = (e.clientY - (r.top + r.height / 2)) / r.height;
      desk.style.transform = `perspective(1200px) rotateY(${(x * 4).toFixed(2)}deg) rotateX(${(-y * 3).toFixed(2)}deg)`;
    });
    hero.addEventListener("pointerleave", () => { desk.style.transform = ""; });
  }

  // ---------------------------------------------------------------- Get started: ink wipe, then the lab
  $$("[data-start]").forEach((a) => a.addEventListener("click", (e) => {
    if (reduce || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    document.body.classList.add("leaving");
    setTimeout(() => { location.href = a.getAttribute("href"); }, 520);
  }));
  addEventListener("pageshow", () => document.body.classList.remove("leaving"));

  // ---------------------------------------------------------------- 🖐 Touch toy
  const cv = $("#toy");
  if (cv) touchToy(cv);

  function touchToy(cv) {
    const ctx = cv.getContext("2d");
    const meta = $("#toyMeta");
    const W = 1.6, H = 1.0, TABLE = 0.22;           // world: metres; table top at y = TABLE (y up)
    const G = 9.81, SUB = 8, REST = 0.15, MU = 0.5;
    let scale = 1, cw = 0, ch = 0, bodies = [], hand = null, last = 0, running = false;
    const COLORS = { blue: "#2f5bff", orange: "#ff5a1f", yellow: "#ffd23f", green: "#7fd1a4", paper: "#fffdf8" };

    function scene() {
      bodies = [
        box("book", 0.42, TABLE + 0.025, 0.17, 0.025, 0.62, COLORS.blue),
        box("mug", 0.42, TABLE + 0.05 + 0.048, 0.045, 0.048, 0.31, COLORS.orange),
        box("bottle", 0.9, TABLE + 0.12, 0.035, 0.12, 0.52, COLORS.green),
        circ("lemon", 1.12, TABLE + 0.035, 0.035, 0.12, COLORS.yellow),
        circ("ball", 1.32, TABLE + 0.037, 0.037, 0.145, COLORS.paper),
        box("box", 0.2, TABLE + 0.06, 0.06, 0.06, 0.4, "#ffb38f"),
      ];
    }
    function box(id, x, y, hw, hh, m, c) { return { id, kind: "box", x, y, hw, hh, vx: 0, vy: 0, m, im: 1 / m, c, f: 0, cp: null }; }
    function circ(id, x, y, r, m, c) { return { id, kind: "circ", x, y, r, vx: 0, vy: 0, m, im: 1 / m, c, f: 0, cp: null }; }

    // the hand: three fingertip capsules stacked under the pointer + a palm, kinematic (infinite mass)
    function handParts(px, py) {
      return [
        { kind: "circ", x: px, y: py + 0.03, r: 0.026, im: 0, hand: true },
        { kind: "circ", x: px + 0.006, y: py - 0.012, r: 0.022, im: 0, hand: true },
        { kind: "circ", x: px + 0.01, y: py - 0.05, r: 0.02, im: 0, hand: true },
      ];
    }

    function resize() {
      const r = cv.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
      cw = r.width; ch = r.height;
      cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      scale = Math.min(cw / W, ch / H);
      if (!running) draw();
    }
    const toPx = (x, y) => [x * scale + (cw - W * scale) / 2, ch - y * scale];
    const toW = (sx, sy) => [(sx - (cw - W * scale) / 2) / scale, (ch - sy) / scale];

    // pointer -> hand target with velocity
    let target = null, prev = null;
    cv.addEventListener("pointermove", (e) => {
      const r = cv.getBoundingClientRect();
      let [x, y] = toW(e.clientX - r.left, e.clientY - r.top);
      y = Math.max(TABLE + 0.072, y);                 // fingertips stop at the table
      x = Math.min(W - 0.03, Math.max(0.03, x));
      target = { x, y, t: performance.now() };
      wake();
    });
    cv.addEventListener("pointerleave", () => { target = null; hand = null; prev = null; });
    cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture?.(e.pointerId); });
    $("#toyReset")?.addEventListener("click", () => { scene(); wake(); });

    // --- collisions: each returns {nx, ny, depth, px, py} with n pointing from a to b
    function collide(a, b) {
      if (a.kind === "circ" && b.kind === "circ") {
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy), s = a.r + b.r;
        if (d >= s || d === 0) return null;
        return { nx: dx / d, ny: dy / d, depth: s - d, px: a.x + dx / d * a.r, py: a.y + dy / d * a.r };
      }
      if (a.kind === "box" && b.kind === "box") {
        const ox = a.hw + b.hw - Math.abs(b.x - a.x), oy = a.hh + b.hh - Math.abs(b.y - a.y);
        if (ox <= 0 || oy <= 0) return null;
        if (ox < oy) { const sx = Math.sign(b.x - a.x) || 1; return { nx: sx, ny: 0, depth: ox, px: a.x + sx * a.hw, py: (Math.max(a.y - a.hh, b.y - b.hh) + Math.min(a.y + a.hh, b.y + b.hh)) / 2 }; }
        const sy = Math.sign(b.y - a.y) || 1; return { nx: 0, ny: sy, depth: oy, px: (Math.max(a.x - a.hw, b.x - b.hw) + Math.min(a.x + a.hw, b.x + b.hw)) / 2, py: a.y + sy * a.hh };
      }
      const flip = a.kind === "circ";                  // box vs circle
      const bx = flip ? b : a, c = flip ? a : b;
      const qx = Math.max(bx.x - bx.hw, Math.min(c.x, bx.x + bx.hw)), qy = Math.max(bx.y - bx.hh, Math.min(c.y, bx.y + bx.hh));
      let dx = c.x - qx, dy = c.y - qy, d = Math.hypot(dx, dy), nx, ny, depth;
      if (d > 0) { if (d >= c.r) return null; nx = dx / d; ny = dy / d; depth = c.r - d; }
      else {                                            // centre inside the box: push out along the shallow axis
        const ox = bx.hw - Math.abs(c.x - bx.x), oy = bx.hh - Math.abs(c.y - bx.y);
        if (ox < oy) { nx = Math.sign(c.x - bx.x) || 1; ny = 0; depth = ox + c.r; } else { nx = 0; ny = Math.sign(c.y - bx.y) || 1; depth = oy + c.r; }
      }
      return flip ? { nx: -nx, ny: -ny, depth, px: qx, py: qy } : { nx, ny, depth, px: qx, py: qy };
    }

    function resolve(a, b, k, dt, va, vb) {
      const ims = a.im + b.im;
      if (!ims) return 0;
      const rvx = (vb ? vb.vx : b.vx) - (va ? va.vx : a.vx), rvy = (vb ? vb.vy : b.vy) - (va ? va.vy : a.vy);
      const vn = rvx * k.nx + rvy * k.ny;
      // positional correction (split by inverse mass)
      const corr = Math.max(0, k.depth - 0.0005) / ims * 0.8;
      a.x -= k.nx * corr * a.im; a.y -= k.ny * corr * a.im; b.x += k.nx * corr * b.im; b.y += k.ny * corr * b.im;
      if (vn > 0) return 0;
      const jn = -(1 + REST) * vn / ims;
      // friction along the tangent, Coulomb-limited
      const tx = -k.ny, ty = k.nx, vt = rvx * tx + rvy * ty;
      const jt = Math.max(-MU * jn, Math.min(MU * jn, -vt / ims));
      const ix = k.nx * jn + tx * jt, iy = k.ny * jn + ty * jt;
      a.vx -= ix * a.im; a.vy -= iy * a.im; b.vx += ix * b.im; b.vy += iy * b.im;
      return jn;
    }

    function step(dt) {
      // hand kinematics: velocity from the pointer, moved toward it this substep
      if (target) {
        const hx = hand ? hand.x : target.x, hy = hand ? hand.y : target.y;
        const vx = (target.x - hx) / Math.max(dt * SUB, 1e-3) * 0.35, vy = (target.y - hy) / Math.max(dt * SUB, 1e-3) * 0.35;
        const nx = hx + vx * dt, ny = hy + vy * dt;
        hand = { x: nx, y: ny, vx, vy, parts: handParts(nx, ny) };
      }
      for (const b of bodies) {
        b.vy -= G * dt;
        b.x += b.vx * dt; b.y += b.vy * dt;
      }
      for (let it = 0; it < 3; it++) {
        for (const b of bodies) {                       // table top and the ends of the table
          const bot = b.kind === "box" ? b.hh : b.r, side = b.kind === "box" ? b.hw : b.r;
          if (b.y - bot < TABLE) {
            b.y = TABLE + bot;
            if (b.vy < 0) {
              const jn = -(1 + REST) * b.vy * b.m;
              b.vy = -REST * b.vy;
              const jt = Math.min(MU * jn, Math.abs(b.vx) * b.m);
              b.vx -= Math.sign(b.vx) * jt / b.m;
              if (b.kind === "circ") b.vx *= 0.999;    // rolling: barely slowed
            }
          }
          if (b.x - side < 0.02) { b.x = 0.02 + side; b.vx = Math.abs(b.vx) * 0.3; }
          if (b.x + side > W - 0.02) { b.x = W - 0.02 - side; b.vx = -Math.abs(b.vx) * 0.3; }
        }
        for (let i = 0; i < bodies.length; i++)
          for (let j = i + 1; j < bodies.length; j++) {
            const k = collide(bodies[i], bodies[j]);
            if (k) resolve(bodies[i], bodies[j], k, dt);
          }
        if (hand)
          for (const p of hand.parts)
            for (const b of bodies) {
              const k = collide(p, b);
              if (!k) continue;
              const jn = resolve(p, b, k, dt, hand, null);
              const f = jn / dt / 3;                    // impulse over the substep -> newtons (3 iterations)
              if (f > b.f) { b.f = f; b.cp = [k.px, k.py]; }
            }
      }
    }

    function draw() {
      ctx.clearRect(0, 0, cw, ch);
      // grid paper + table
      ctx.strokeStyle = "#ece6d8"; ctx.lineWidth = 1;
      for (let x = 0; x < cw; x += 24) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, ch); ctx.stroke(); }
      for (let y = 0; y < ch; y += 24) { ctx.beginPath(); ctx.moveTo(0, y + .5); ctx.lineTo(cw, y + .5); ctx.stroke(); }
      const [tx0, ty] = toPx(0, TABLE), [tx1] = toPx(W, TABLE);
      ctx.fillStyle = "#d9b382"; ctx.fillRect(tx0, ty, tx1 - tx0, ch - ty);
      ctx.strokeStyle = "#c99d68"; ctx.lineWidth = 1.2;
      for (let y = ty + 10; y < ch; y += 14) { ctx.beginPath(); ctx.moveTo(tx0, y); ctx.bezierCurveTo(tx0 + 80, y - 4, tx1 - 80, y + 4, tx1, y); ctx.stroke(); }
      ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(tx0, ty); ctx.lineTo(tx1, ty); ctx.stroke();
      // bodies
      for (const b of bodies) {
        ctx.fillStyle = b.c; ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 2;
        if (b.kind === "box") {
          const [x0, y0] = toPx(b.x - b.hw, b.y + b.hh);
          roundRect(x0, y0, 2 * b.hw * scale, 2 * b.hh * scale, 4); ctx.fill(); ctx.stroke();
        } else {
          const [x, y] = toPx(b.x, b.y);
          ctx.beginPath(); ctx.arc(x, y, b.r * scale, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        }
        if (scale < 320) continue;                     // too small to label without overlaps (phones)
        const [lx, ly] = toPx(b.x, b.y + (b.kind === "box" ? b.hh : b.r));
        ctx.fillStyle = "#5f5a50"; ctx.font = "500 11px 'IBM Plex Mono', monospace"; ctx.textAlign = "center";
        ctx.fillText(b.id, lx, ly - 6);
      }
      // contact markers (force in N), as in the app
      const touching = [];
      for (const b of bodies) {
        if (b.f > 0.05 && b.cp) {
          touching.push(`${b.id} ${b.f.toFixed(1)} N`);
          const [x, y] = toPx(b.cp[0], b.cp[1]), r = Math.min(26, 7 + 4 * Math.sqrt(b.f));
          ctx.fillStyle = "rgba(255,90,31,.22)"; ctx.strokeStyle = "#ff5a1f"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          ctx.fillStyle = "#1e2622"; ctx.font = "600 12px 'IBM Plex Mono', monospace"; ctx.textAlign = "left";
          ctx.fillText(`${b.f.toFixed(1)} N`, x + r + 4, y - 4);
        }
      }
      // the hand: bones + joints, like the app's skeleton overlay
      if (hand) {
        const ps = hand.parts.map((p) => toPx(p.x, p.y)), [wx, wy] = toPx(hand.x + 0.09, hand.y + 0.16);
        ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 3; ctx.lineCap = "round";
        ps.forEach(([x, y]) => { ctx.beginPath(); ctx.moveTo(wx, wy); ctx.lineTo(x, y); ctx.stroke(); });
        hand.parts.forEach((p, i) => {
          const [x, y] = ps[i];
          ctx.fillStyle = "#2f5bff"; ctx.globalAlpha = .18; ctx.beginPath(); ctx.arc(x, y, p.r * scale, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
          ctx.fillStyle = "#fffdf8"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        });
        ctx.fillStyle = "#1e2622"; ctx.beginPath(); ctx.arc(wx, wy, 6, 0, Math.PI * 2); ctx.fill();
      }
      if (meta) meta.textContent = touching.length ? `touching ${touching.join(" · ")}` : hand ? "push something" : "move over the table";
    }
    function roundRect(x, y, w, h, r) {
      ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }

    let idle = 0;
    function frame(now) {
      const dt = Math.min(1 / 30, (now - last) / 1000 || 1 / 60); last = now;
      bodies.forEach((b) => { b.f = 0; b.cp = null; });
      for (let i = 0; i < SUB; i++) step(dt / SUB);
      draw();
      const moving = bodies.some((b) => Math.abs(b.vx) + Math.abs(b.vy) > 0.004) || target;
      idle = moving ? 0 : idle + 1;
      if (idle > 30 || !visible) { running = false; return; }
      requestAnimationFrame(frame);
    }
    function wake() { if (!running && visible) { running = true; last = performance.now(); idle = 0; requestAnimationFrame(frame); } }
    let visible = false;
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) wake(); }).observe(cv);
    scene();
    addEventListener("resize", resize);
    resize();
  }

  // ---------------------------------------------------------------- walk-around scan animation
  const orbit = $("#orbitSvg");
  if (orbit) orbitAnim(orbit);

  function orbitAnim(svg) {
    const NS = "http://www.w3.org/2000/svg";
    const phone = $("#orbitPhone", svg), cams = $("#orbitCams", svg), pts = $("#sfmPts", svg), label = $("#orbitStep", svg);
    const carves = $$(".carve", svg);
    const CX = 210, CY = 190, RX = 170, RY = 62, LOOP = 9000;
    // feature points on the object surfaces and table (fixed, so the "solve" looks consistent)
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const surf = [];
    for (let i = 0; i < 26; i++) surf.push([120 + rnd() * 70, 170 + rnd() * 30]);
    for (let i = 0; i < 22; i++) surf.push([220 + rnd() * 40, 140 + rnd() * 60]);
    for (let i = 0; i < 12; i++) { const a = rnd() * 6.28; surf.push([300 + Math.cos(a) * 18 * rnd(), 190 + Math.sin(a) * 12 * rnd()]); }
    for (let i = 0; i < 30; i++) surf.push([70 + rnd() * 280, 205 + rnd() * 40]);
    const steps = ["1 · walk around with your phone", "2 · solve every camera position", "3 · find what rises above the table", "4 · carve real 3D shapes"];
    let t0 = 0, raf = 0, shownCams = 0, shownPts = 0, stage = -1;
    const show = (k) => { if (k !== stage) { stage = k; label.textContent = steps[k]; } };
    function reset() {
      cams.replaceChildren(); pts.replaceChildren(); shownCams = 0; shownPts = 0;
      carves.forEach((c) => c.classList.remove("on"));
    }
    function at(a) { return [CX + RX * Math.cos(a), CY + RY * Math.sin(a)]; }
    function tick(now) {
      if (!t0) t0 = now;
      const u = ((now - t0) % LOOP) / LOOP;
      if (u < 0.01 && shownCams > 0) reset();
      const lap = Math.min(1, u / 0.62), a = Math.PI * 0.5 + lap * Math.PI * 2;
      const [x, y] = at(a), s = 0.75 + 0.35 * ((y - (CY - RY)) / (2 * RY));
      phone.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${s.toFixed(2)})`);
      phone.style.opacity = u < 0.66 ? 1 : Math.max(0, 1 - (u - 0.66) * 12);
      const wantCams = Math.floor(lap * 14);
      while (shownCams < wantCams && shownCams < 14) {   // a camera frustum where each keyframe was taken
        const [cx, cy] = at(Math.PI * 0.5 + (shownCams / 14) * Math.PI * 2);
        const dx = CX - cx, dy = CY - 10 - cy, d = Math.hypot(dx, dy), ux = dx / d, uy = dy / d;
        const p = document.createElementNS(NS, "path");
        p.setAttribute("d", `M${cx} ${cy} L${cx + ux * 16 - uy * 6} ${cy + uy * 16 + ux * 6} L${cx + ux * 16 + uy * 6} ${cy + uy * 16 - ux * 6} Z`);
        cams.appendChild(p); shownCams++;
      }
      const wantPts = Math.floor(Math.min(1, u / 0.5) * surf.length);
      while (shownPts < wantPts) {
        const c = document.createElementNS(NS, "circle");
        c.setAttribute("cx", surf[shownPts][0].toFixed(1)); c.setAttribute("cy", surf[shownPts][1].toFixed(1)); c.setAttribute("r", "1.6");
        pts.appendChild(c); shownPts++;
      }
      show(u < 0.3 ? 0 : u < 0.62 ? 1 : u < 0.72 ? 2 : 3);
      if (u > 0.68) carves[0].classList.add("on");
      if (u > 0.72) carves[1].classList.add("on");
      if (u > 0.76) carves[2].classList.add("on");
      pts.style.opacity = u > 0.8 ? Math.max(0.15, 1 - (u - 0.8) * 5) : 1;
      raf = requestAnimationFrame(tick);
    }
    if (reduce) {                                      // final state, no motion
      label.textContent = steps[3]; carves.forEach((c) => c.classList.add("on")); phone.style.opacity = 0;
      return;
    }
    new IntersectionObserver((es) => {
      if (es[0].isIntersecting) { if (!raf) raf = requestAnimationFrame(tick); }
      else { cancelAnimationFrame(raf); raf = 0; }
    }).observe(svg);
  }
})();

// ---- demo videos: two tabs, play only while on screen
(() => {
  const root = document.querySelector("[data-demo]");
  if (!root) return;
  const vids = [...root.querySelectorAll("video")], caps = [...root.querySelectorAll(".demo-cap")];
  const tabs = [...root.querySelectorAll(".demo-tabs button")], btn = root.querySelector(".demo-play");
  let cur = "real", visible = false, paused = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sync = () => {
    vids.forEach((v) => {
      const on = v.dataset.v === cur;
      v.classList.toggle("on", on);
      if (on && visible && !paused) { v.preload = "auto"; v.play().catch(() => {}); } else v.pause();
    });
    caps.forEach((c) => { c.hidden = c.dataset.v !== cur; });
    tabs.forEach((t) => { const on = t.dataset.v === cur; t.classList.toggle("on", on); t.setAttribute("aria-selected", on); });
    btn.textContent = paused ? "▶" : "❚❚"; btn.setAttribute("aria-label", paused ? "Play video" : "Pause video");
  };
  tabs.forEach((t) => t.addEventListener("click", () => { cur = t.dataset.v; const v = vids.find((x) => x.dataset.v === cur); if (v) v.currentTime = 0; sync(); }));
  btn.addEventListener("click", () => { paused = !paused; sync(); });
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; sync(); }, { threshold: 0.35 }).observe(root.querySelector(".demo-frame"));
  sync();
})();
