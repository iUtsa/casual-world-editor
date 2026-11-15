/* Causal World Editor landing page. No dependencies. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fmt = (x, d = 2) => Number(x).toFixed(d);

  /* ---------- toast + copy ---------- */
  const toast = $("#toast");
  let toastTimer;
  function say(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
  }
  $$(".copy").forEach((b) =>
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; say("Copied to clipboard"); }
      catch { say("Couldn't copy. Select the text instead."); }
      setTimeout(() => (b.textContent = "Copy"), 1400);
    })
  );

  /* ---------- tween helper ---------- */
  function tween(dur, step, ease = (t) => t) {
    return new Promise((res) => {
      if (reduceMotion) { step(1); return res(); }
      const t0 = performance.now();
      const tick = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        step(ease(t));
        t < 1 ? requestAnimationFrame(tick) : res();
      };
      requestAnimationFrame(tick);
    });
  }
  const easeIn = (t) => t * t;
  const easeOut = (t) => 1 - (1 - t) * (1 - t);

  /* ---------- hero desk demo ---------- */
  const desk = {
    g: 9.81, heavy: false, toppled: false, busy: false,
    bottle: $("#bottle"), ghost: $("#bottleGhost"), arrow: $("#pushArrow"), ball: $("#ball"),
    answer: $("#deskAnswer"), status: $("#deskStatus"), gText: $("#gReadout"), bookText: $("#bookMass"),
  };
  const bookMass = () => (desk.heavy ? 6.2 : 0.62);
  function flash(el) { el.classList.remove("flash"); void el.getBBox(); el.classList.add("flash"); fitTags(); }
  // size each label box to its text once fonts are in
  function fitTags() {
    $$("#deskSvg #labels > g, #deskSvg g:has(> #gReadout)").forEach((g) => {
      const rect = g.querySelector("rect");
      const w = Math.max(...$$("text", g).map((t) => t.getComputedTextLength()));
      if (w > 0) rect.setAttribute("width", String(Math.ceil(w + 20)));
    });
  }
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(fitTags);
  function setAnswer(html) { desk.answer.innerHTML = html; }

  async function knock() {
    if (desk.toppled) { setAnswer("The bottle is already lying down. Press <b>undo all</b> to stand it back up."); return; }
    desk.status.textContent = "simulating · push 1.2 m/s →";
    desk.ghost.setAttribute("opacity", "1");
    await tween(300, (t) => desk.arrow.setAttribute("opacity", String(t)));
    // tipping accelerates like a falling rod, then a small bounce on impact
    const slow = Math.sqrt(9.81 / desk.g);
    await tween(700 * slow, (t) => desk.bottle.setAttribute("transform", `rotate(${88 * t} 512 312)`), easeIn);
    await tween(180, (t) => desk.bottle.setAttribute("transform", `rotate(${88 - 5 * Math.sin(Math.PI * t)} 512 312)`));
    desk.arrow.setAttribute("opacity", "0");
    desk.toppled = true;
    const tip = 0.42 * slow;
    setAnswer(`The twin tips over to the right after <b>${fmt(tip)} s</b>. The real bottle (dashed) never moved, so nothing spilled.`);
    desk.status.textContent = "settled · checkpoint saved";
  }

  async function drop() {
    desk.status.textContent = "simulating · baseball 0.145 kg";
    const h = 0.4; // metres above the book
    const x = 318, top = 30, surface = 282 - 13;
    const pxPerM = (surface - top) / h;
    const slowmo = 3; // play 3x slower than real time
    const g = desk.g;
    desk.ball.setAttribute("opacity", "1");
    // fall + two bounces with restitution 0.45, all from kinematics
    let y = 0, v = 0, bounces = 0;
    const tHit = Math.sqrt((2 * h) / g), vHit = Math.sqrt(2 * g * h);
    await new Promise((res) => {
      let last = performance.now();
      const tick = (now) => {
        let dt = Math.min(0.05, (now - last) / 1000) / slowmo; last = now;
        if (reduceMotion) dt = 10;
        v += g * dt; y += v * dt;
        if (y >= h) { y = h; v = -0.45 * v; bounces++; }
        desk.ball.setAttribute("transform", `translate(${x} ${top + y * pxPerM})`);
        if (bounces >= 3 || reduceMotion) { desk.ball.setAttribute("transform", `translate(${x} ${surface})`); res(); return; }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    flash(desk.bookText);
    const slide = desk.heavy ? "The 6.2 kg book doesn't budge." : "The book shifts a few millimetres; the mug stays on it.";
    setAnswer(`Dropped from 40 cm, it lands after <b>${fmt(tHit)} s</b> at <b>${fmt(vHit)} m/s</b>. ${slide}`);
    desk.status.textContent = "settled · checkpoint saved";
  }

  function heavy() {
    desk.heavy = !desk.heavy;
    desk.bookText.textContent = `book · 2.4 cm · ${desk.heavy ? "6.2" : "0.62"} kg`;
    flash(desk.bookText);
    const holdN = 0.38 * (bookMass() + 0.31) * desk.g;
    setAnswer(desk.heavy
      ? `The book is now <b>6.2 kg</b>. To slide it, with the mug on top, you'd need about <b>${fmt(holdN, 1)} N</b> (μ × m × g). A desk-sized nudge won't do it.`
      : `Back to <b>0.62 kg</b>. About <b>${fmt(holdN, 1)} N</b> is enough to slide it now.`);
  }

  function moon() {
    const onMoon = desk.g === 9.81;
    desk.g = onMoon ? 1.62 : 9.81;
    desk.gText.textContent = `g = ${fmt(desk.g)} m/s²`;
    flash(desk.gText);
    setAnswer(onMoon
      ? `Gravity is now <b>1.62 m/s²</b>. The mug presses on the book with <b>${fmt(0.31 * 1.62)} N</b> instead of ${fmt(0.31 * 9.81)} N. Try dropping the baseball again.`
      : `Back on Earth: <b>9.81 m/s²</b>.`);
  }

  function resetDesk() {
    desk.g = 9.81; desk.heavy = false; desk.toppled = false;
    desk.bottle.removeAttribute("transform");
    desk.ghost.setAttribute("opacity", "0");
    desk.arrow.setAttribute("opacity", "0");
    desk.ball.setAttribute("opacity", "0");
    desk.gText.textContent = "g = 9.81 m/s²";
    desk.bookText.textContent = "book · 2.4 cm · 0.62 kg";
    desk.status.textContent = "scanned · 4 objects · table plane found";
    setAnswer("Scene restored to the original scan.");
  }

  const acts = { knock, drop, heavy, moon, reset: resetDesk };
  $$("#deskChips .chip").forEach((b) =>
    b.addEventListener("click", async () => {
      if (desk.busy) return;
      desk.busy = true;
      $$("#deskChips .chip").forEach((c) => c.classList.toggle("active", c === b && b.dataset.act !== "reset"));
      try { await acts[b.dataset.act](); } finally { desk.busy = false; }
    })
  );

  /* ---------- ramp simulation ---------- */
  const DEFAULTS = { angle: 30, mass: 2, mu: 0.2, length: 1, g: 9.81 };
  let params = { ...DEFAULTS };
  let run = null; // { traj, result, start }
  const canvas = $("#sim");
  const ctx = canvas.getContext("2d");
  const card = $("#readingCard");
  const slowBox = $("#slowmo");

  function simulate(p) {
    const th = (p.angle * Math.PI) / 180;
    const sin = Math.sin(th), cos = Math.cos(th);
    const aTheory = p.g * (sin - p.mu * cos);
    const stuck = Math.tan(th) <= p.mu + 1e-9;
    const dt = 0.001;
    const traj = [{ t: 0, s: 0, v: 0 }];
    let s = 0, v = 0, t = 0, tCross = null, vCross = null;
    if (!stuck) {
      // semi-implicit Euler along the incline, forces evaluated each step
      while (t < 30) {
        const normal = p.mass * p.g * cos;
        const net = p.mass * p.g * sin - p.mu * normal; // kinetic friction opposes motion
        const a = net / p.mass;
        const v1 = v + a * dt;
        const s1 = s + v1 * dt;
        t += dt;
        if (s1 >= p.length) {
          const f = (p.length - s) / (s1 - s); // interpolate the crossing
          tCross = t - dt + f * dt;
          vCross = v + f * (v1 - v);
          traj.push({ t: tCross, s: p.length, v: vCross });
          break;
        }
        s = s1; v = v1;
        traj.push({ t, s, v });
      }
    }
    const res = { stuck, aTheory, th };
    if (!stuck) {
      res.mT = tCross; res.pT = Math.sqrt((2 * p.length) / aTheory);
      res.mV = vCross; res.pV = Math.sqrt(2 * aTheory * p.length);
      res.mE = 0.5 * p.mass * vCross * vCross;
      res.pE = p.mass * p.g * p.length * sin - p.mu * p.mass * p.g * cos * p.length;
      const rel = (m, q) => 1 - Math.abs(m - q) / Math.abs(q);
      res.agree = ((rel(res.mT, res.pT) + rel(res.mV, res.pV) + rel(res.mE, res.pE)) / 3) * 100;
    }
    return { traj, res };
  }

  function insightFor(p, r, q, rq) {
    if (r.stuck) {
      return `It doesn't move. tan θ = <b>${fmt(Math.tan(r.th))}</b> is not bigger than μ = <b>${fmt(p.mu)}</b>, so static friction holds the block in place. Try a steeper angle.`;
    }
    const same = q && Object.keys(p).every((k) => Math.abs(p[k] - q[k]) < 1e-9);
    if (!q || !rq || rq.stuck || same) {
      return `It reaches the bottom in <b>${fmt(r.mT)} s</b> at <b>${fmt(r.mV)} m/s</b>. The ghost blocks mark every 0.1 s: the gaps grow because it speeds up at a constant <b>${fmt(r.aTheory)} m/s²</b>.`;
    }
    const changed = Object.keys(p).filter((k) => Math.abs(p[k] - q[k]) > 1e-9);
    if (changed.length === 1 && changed[0] === "mass") {
      return `Same time as before, <b>${fmt(r.mT)} s</b>. Mass cancels: a = g(sin θ − μ cos θ) has no m in it. The ${p.mass > q.mass ? "heavier" : "lighter"} block carries ${p.mass > q.mass ? "more" : "less"} energy (${fmt(r.mE, 1)} J), not more speed.`;
    }
    if (changed.length === 1 && changed[0] === "g") {
      const ratio = r.mT / rq.mT;
      return `Time scales with 1/√g. It takes <b>${fmt(r.mT)} s</b>, ${fmt(ratio > 1 ? ratio : 1 / ratio)}× ${ratio > 1 ? "longer" : "shorter"} than before.`;
    }
    if (changed.length === 1 && changed[0] === "mu") {
      return `Friction changed the acceleration from ${fmt(rq.aTheory)} to <b>${fmt(r.aTheory)} m/s²</b>, so it arrives in <b>${fmt(r.mT)} s</b> instead of ${fmt(rq.mT)} s.`;
    }
    if (changed.length === 1 && changed[0] === "angle") {
      return `${p.angle > q.angle ? "Steeper" : "Gentler"} ramp: acceleration goes from ${fmt(rq.aTheory)} to <b>${fmt(r.aTheory)} m/s²</b>, and it arrives in <b>${fmt(r.mT)} s</b> instead of ${fmt(rq.mT)} s.`;
    }
    return `Acceleration <b>${fmt(r.aTheory)} m/s²</b>, bottom reached in <b>${fmt(r.mT)} s</b> at <b>${fmt(r.mV)} m/s</b>.`;
  }

  function showCheck(p, r) {
    const pill = $("#checkPill");
    const agree = $("#agree").parentElement;
    if (r.stuck) {
      $("#mT").textContent = "never"; $("#pT").textContent = "never";
      $("#mV").textContent = "0.00 m/s"; $("#pV").textContent = "0.00 m/s";
      $("#mE").textContent = "0.00 J"; $("#pE").textContent = "0.00 J";
      $("#agree").textContent = "stays put, as predicted";
      agree.classList.add("na");
      pill.className = "pill ok"; pill.textContent = "consistent";
      return;
    }
    agree.classList.remove("na");
    $("#mT").textContent = `${fmt(r.mT, 3)} s`; $("#pT").textContent = `${fmt(r.pT, 3)} s`;
    $("#mV").textContent = `${fmt(r.mV, 3)} m/s`; $("#pV").textContent = `${fmt(r.pV, 3)} m/s`;
    $("#mE").textContent = `${fmt(r.mE, 2)} J`; $("#pE").textContent = `${fmt(r.pE, 2)} J`;
    $("#agree").textContent = `${fmt(r.agree, 2)}%`;
    pill.className = r.agree > 99 ? "pill ok" : "pill warn";
    pill.textContent = r.agree > 99 ? "passes" : "check";
  }

  function addHistory(p, r) {
    const li = document.createElement("li");
    li.innerHTML = `<span>θ ${p.angle}° · ${p.mass} kg · μ ${fmt(p.mu)} · g ${fmt(p.g)}</span><b>${r.stuck ? "stays" : fmt(r.mT) + " s"}</b>`;
    const h = $("#history");
    h.prepend(li);
    while (h.children.length > 6) h.lastChild.remove();
  }

  function syncCard() {
    for (const k of Object.keys(params)) card.elements[k].value = String(+params[k].toFixed(3));
  }

  function start(p, fresh = false) {
    // compare with the last run started; before any run, compare with the figure as it was read
    const basis = fresh ? null : run ? { p: run.p, res: run.res } : { p: { ...DEFAULTS }, res: simulate(DEFAULTS).res };
    params = { ...p };
    syncCard();
    const out = simulate(params);
    run = { ...out, p: { ...params }, start: performance.now() };
    $("#simMeta").textContent = out.res.stuck ? "static · no motion" : "semi-implicit Euler · dt 1 ms";
    $("#checkPill").className = "pill run"; $("#checkPill").textContent = "running";
    $("#insight").innerHTML = "Simulating…";
    run.done = false;
    run.onDone = () => {
      showCheck(params, out.res);
      $("#insight").innerHTML = insightFor(params, out.res, basis && basis.p, basis && basis.res);
      addHistory(params, out.res);
    };
    if (out.res.stuck) { run.done = true; run.onDone(); }
    else {
      // fallback in case animation frames are paused (hidden tab): finish the check on a timer
      const me = run;
      const ms = (me.traj[me.traj.length - 1].t / (slowBox.checked ? 0.5 : 1)) * 1000 + 120;
      setTimeout(() => { if (run === me && !me.done) { me.done = true; me.onDone(); draw(); } }, ms);
    }
    draw();
  }

  /* drawing */
  let W = 0, H = 0;
  function resize() {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function grid() {
    ctx.fillStyle = "#fffdf8"; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(20,20,20,.06)"; ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 20) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 20) { ctx.beginPath(); ctx.moveTo(0, y + .5); ctx.lineTo(W, y + .5); ctx.stroke(); }
  }

  function geometry(p) {
    const th = (p.angle * Math.PI) / 180;
    const compact = W < 520;
    const aw = W * (compact ? 0.86 : 0.58), ah = H * (compact ? 0.5 : 0.7);
    const run = p.length * Math.cos(th), rise = p.length * Math.sin(th);
    const k = Math.min(aw / run, ah / rise);
    const x0 = compact ? W * 0.07 : W * 0.06, yb = H - (compact ? 34 : 40);
    return { th, k, len: p.length, top: { x: x0, y: yb - rise * k }, bottom: { x: x0 + run * k, y: yb }, corner: { x: x0, y: yb } };
  }
  // keep the whole block on the ramp: its centre runs from size/2 to (ramp - size/2)
  function along(gm, s, size) {
    const px = gm.len * gm.k;
    const d = size / 2 + (s / gm.len) * Math.max(0, px - size);
    return { x: gm.top.x + d * Math.cos(gm.th), y: gm.top.y + d * Math.sin(gm.th) };
  }

  function block(gm, s, size, alpha, fill) {
    const { x: cx, y: cy } = along(gm, s, size);
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(gm.th);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fill; ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.rect(-size / 2, -size, size, size); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  function draw(now = performance.now()) {
    if (!W) return;
    const p = run ? run.p : params;
    grid();
    const gm = geometry(p);
    const { top, bottom, corner, th } = gm;
    // ground
    ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(8, corner.y); ctx.lineTo(W - 8, corner.y); ctx.stroke();
    // ramp with hatch
    ctx.save();
    ctx.beginPath(); ctx.moveTo(top.x, top.y); ctx.lineTo(bottom.x, bottom.y); ctx.lineTo(corner.x, corner.y); ctx.closePath();
    ctx.fillStyle = "#f3efe5"; ctx.fill();
    ctx.clip();
    ctx.strokeStyle = "rgba(20,20,20,.18)"; ctx.lineWidth = 1;
    for (let i = -H; i < W; i += 9) { ctx.beginPath(); ctx.moveTo(i, corner.y); ctx.lineTo(i + H, corner.y - H); ctx.stroke(); }
    ctx.restore();
    ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(top.x, top.y); ctx.lineTo(bottom.x, bottom.y); ctx.lineTo(corner.x, corner.y); ctx.closePath(); ctx.stroke();
    // angle arc
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(bottom.x, bottom.y, 34, Math.PI, Math.PI + th); ctx.stroke();
    ctx.fillStyle = "#1e2622"; ctx.font = "500 13px 'IBM Plex Mono', monospace";
    ctx.fillText(`θ ${p.angle}°`, bottom.x - 58, bottom.y + 19);

    // playback state
    const size = Math.max(16, Math.min(56, 26 * Math.cbrt(p.mass / 2)));
    let tNow = 0, s = 0, v = 0;
    const traj = run ? run.traj : [{ t: 0, s: 0, v: 0 }];
    const tEnd = traj[traj.length - 1].t;
    if (run && !run.res.stuck) {
      const speed = slowBox.checked ? 0.5 : 1;
      tNow = Math.min(tEnd, ((now - run.start) / 1000) * speed);
      // find sample
      const i = Math.min(traj.length - 1, Math.max(0, Math.round(tNow / 0.001)));
      ({ s, v } = traj[i]);
      if (tNow >= tEnd) { s = traj[traj.length - 1].s; v = traj[traj.length - 1].v; }
      // strobe ghosts every 0.1 s
      for (let tg = 0; tg < tNow - 1e-6; tg += 0.1) {
        const j = Math.min(traj.length - 1, Math.round(tg / 0.001));
        block(gm, traj[j].s, size, 0.22, "#ff5a1f");
      }
    }
    block(gm, s, size, 1, "#ff5a1f");
    // mass label on block
    ctx.save();
    const b = along(gm, s, size);
    ctx.translate(b.x, b.y); ctx.rotate(th);
    ctx.fillStyle = "#fff"; ctx.font = `600 ${Math.max(10, Math.min(14, size / 3))}px 'IBM Plex Mono', monospace`; ctx.textAlign = "center";
    if (size >= 24) ctx.fillText(`${+p.mass.toFixed(1)}kg`, 0, -size / 2 + 4);
    ctx.restore();

    // readouts
    ctx.textAlign = "right"; ctx.fillStyle = "#1e2622"; ctx.font = "500 13px 'IBM Plex Mono', monospace";
    const line = W < 520 ? `t ${fmt(tNow)} s  v ${fmt(v)} m/s` : `t ${fmt(tNow)} s   v ${fmt(v)} m/s   μ ${fmt(p.mu)}   g ${fmt(p.g)}`;
    ctx.fillText(line, W - 16, 16 + (W < 520 ? H * 0.42 : H * 0.36) + 22);
    ctx.textAlign = "left";

    // speed chart
    chart(p, traj, tNow, tEnd);

    if (run && run.res.stuck) {
      ctx.fillStyle = "#a5560c"; ctx.font = "600 14px 'Instrument Sans', sans-serif";
      ctx.fillText("static friction holds it", top.x + 8, top.y - 46);
    }

    if (run && !run.done && tNow >= tEnd) { run.done = true; run.onDone(); }
    if (run && !run.done) requestAnimationFrame(draw);
  }

  function chart(p, traj, tNow, tEnd) {
    const compact = W < 520;
    const cw = compact ? W * 0.44 : W * 0.32, ch = compact ? H * 0.42 : H * 0.36;
    const cx = W - cw - 16, cy = 16;
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#1e2622"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.rect(cx, cy, cw, ch); ctx.fill(); ctx.stroke();
    ctx.font = "500 11px 'IBM Plex Mono', monospace"; ctx.fillStyle = "#5f5a50";
    ctx.fillText("speed vs time", cx + 8, cy + 15);
    if (!run || run.res.stuck) return;
    const r = run.res;
    const vMax = Math.max(r.pV, 0.01) * 1.1, tMax = Math.max(r.pT, 0.01) * 1.05;
    const px = (t) => cx + 8 + (t / tMax) * (cw - 16);
    const py = (v) => cy + ch - 8 - (v / vMax) * (ch - 58);
    // theory line
    ctx.strokeStyle = "#2f5bff"; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(r.pT), py(r.pV)); ctx.stroke(); ctx.setLineDash([]);
    // measured dots every 0.05 s
    ctx.fillStyle = "#ff5a1f";
    for (let t = 0; t <= Math.min(tNow, tEnd) + 1e-9; t += Math.max(0.02, tEnd / 24)) {
      const j = Math.min(traj.length - 1, Math.round(t / 0.001));
      ctx.beginPath(); ctx.arc(px(traj[j].t), py(traj[j].v), 2.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.font = "500 10.5px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#2f5bff"; ctx.fillText("- - textbook", cx + 8, cy + 30);
    ctx.fillStyle = "#ff5a1f"; ctx.fillText("••• measured", cx + 8, cy + 43);
  }

  card.addEventListener("submit", (e) => {
    e.preventDefault();
    const next = {};
    for (const k of Object.keys(DEFAULTS)) {
      const el = card.elements[k];
      let val = parseFloat(el.value);
      if (!Number.isFinite(val)) val = DEFAULTS[k];
      val = Math.min(parseFloat(el.max), Math.max(parseFloat(el.min), val));
      next[k] = val;
    }
    start(next);
  });

  function parseWhatIf(text) {
    const q = text.toLowerCase().replace(/,/g, " ");
    if (/\breset\b|start over|original/.test(q)) return { ...DEFAULTS, _reset: true };
    const p = { ...params };
    let hit = false;
    let m;
    if ((m = q.match(/(\d+(?:\.\d+)?)\s*(?:°|deg|degree)/)) || (m = q.match(/angle[^\d]*(\d+(?:\.\d+)?)/))) { p.angle = +m[1]; hit = true; }
    else if (/steeper/.test(q)) { p.angle = params.angle + 15; hit = true; }
    else if (/(gentler|shallower|less steep|flatter)/.test(q)) { p.angle = params.angle - 10; hit = true; }
    if ((m = q.match(/(\d+(?:\.\d+)?)\s*(?:x|×|times)\s*heavier/))) { p.mass = params.mass * +m[1]; hit = true; }
    else if ((m = q.match(/(\d+(?:\.\d+)?)\s*(?:x|×|times)\s*lighter/))) { p.mass = params.mass / +m[1]; hit = true; }
    else if (/twice as heavy|double the mass/.test(q)) { p.mass = params.mass * 2; hit = true; }
    else if (/half (as heavy|the mass)/.test(q)) { p.mass = params.mass / 2; hit = true; }
    else if ((m = q.match(/(\d+(?:\.\d+)?)\s*kg/))) { p.mass = +m[1]; hit = true; }
    if (/frictionless|no friction/.test(q)) { p.mu = 0; hit = true; }
    else if (/\bic(y|e)\b/.test(q)) { p.mu = 0.03; hit = true; }
    else if (/(rough|sandpaper|rubber)/.test(q)) { p.mu = 0.6; hit = true; }
    else if ((m = q.match(/(?:μ|mu|friction)\s*(?:=|of|to|is)?\s*(\d*\.?\d+)/))) { p.mu = +m[1]; hit = true; }
    const planets = { moon: 1.62, mars: 3.71, jupiter: 24.79, earth: 9.81 };
    for (const [k, g] of Object.entries(planets)) if (q.includes(k)) { p.g = g; hit = true; }
    if ((m = q.match(/(\d+(?:\.\d+)?)\s*m\/s/))) { p.g = +m[1]; hit = true; }
    if ((m = q.match(/(\d+(?:\.\d+)?)\s*m\b(?!\/)/)) && /(long|length|ramp|metre|meter)/.test(q)) { p.length = +m[1]; hit = true; }
    if (!hit) return null;
    p.angle = Math.min(80, Math.max(5, p.angle));
    p.mass = Math.min(500, Math.max(0.1, +p.mass.toFixed(3)));
    p.mu = Math.min(1.5, Math.max(0, p.mu));
    p.length = Math.min(5, Math.max(0.2, p.length));
    p.g = Math.min(30, Math.max(0.5, p.g));
    return p;
  }

  function ask(text) {
    const p = parseWhatIf(text);
    if (!p) {
      $("#insight").innerHTML = "This little sketch only understands angle, mass, friction, gravity and ramp length. The app's language layer handles far more, like <b>“two blue whales collide at 10 m/s”</b>.";
      return;
    }
    const fresh = !!p._reset;
    delete p._reset;
    start(p, fresh);
  }
  $("#whatIf").addEventListener("submit", (e) => {
    e.preventDefault();
    const inp = $("#whatIfInput");
    if (inp.value.trim()) ask(inp.value.trim());
    inp.value = "";
  });
  $$("#simChips .chip").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));

  window.addEventListener("resize", resize);
  resize();
  // auto-run the figure once when the section comes into view
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting) && !run) { start({ ...DEFAULTS }); io.disconnect(); }
  }, { threshold: 0.4 });
  io.observe(canvas);

  /* ---------- tabs ---------- */
  const tabs = $$(".tabs [role=tab]");
  function selectTab(tab) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $("#" + t.getAttribute("aria-controls")).hidden = !on;
    });
    if (tab.id === "t-ai") animateLayers();
    if (tab.id === "t-physics") runDominoes();
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(t));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      n.focus(); selectTab(n);
    });
  });

  /* dominoes */
  const NS = "http://www.w3.org/2000/svg";
  const row = $("#dominoRow");
  const dominoes = [];
  for (let i = 0; i < 10; i++) {
    const r = document.createElementNS(NS, "rect");
    r.setAttribute("x", String(46 + i * 34)); r.setAttribute("y", "120");
    r.setAttribute("width", "10"); r.setAttribute("height", "60"); r.setAttribute("rx", "1.5");
    r.setAttribute("fill", i % 2 ? "#fffdf8" : "#2f5bff"); r.setAttribute("stroke", "#1e2622"); r.setAttribute("stroke-width", "2");
    r.classList.add("domino");
    row.appendChild(r); dominoes.push(r);
  }
  let dominoAnims = [];
  function runDominoes() {
    if (reduceMotion || !dominoes[0].animate) return;
    dominoAnims.forEach((a) => a.cancel());
    const loop = 5000;
    dominoAnims = dominoes.map((d, i) => {
      const startAt = 0.18 + i * 0.045, fallDur = 0.06;
      const end = i === dominoes.length - 1 ? 84 : 52;
      return d.animate(
        [
          { transform: "rotate(0deg)", offset: 0 },
          { transform: "rotate(0deg)", offset: startAt },
          { transform: `rotate(${end}deg)`, offset: startAt + fallDur, easing: "ease-in" },
          { transform: `rotate(${end}deg)`, offset: 0.92 },
          { transform: "rotate(0deg)", offset: 1 },
        ],
        { duration: loop, iterations: Infinity }
      );
    });
  }
  runDominoes();

  /* AI lab layers (illustrative) */
  const LAYERS = [
    [" the", 0.04], [",", 0.03], [" same", 0.03], [" edge", 0.04], [" edge", 0.05], [" floor", 0.06],
    [" floor", 0.09], [" floor", 0.12], [" bed", 0.13], [" floor", 0.16], [" floor", 0.19], [" floor", 0.22],
  ];
  const layersEl = $("#layers");
  LAYERS.forEach(([tok, p], i) => {
    const d = document.createElement("div");
    d.className = "layer-row" + (i === LAYERS.length - 1 ? " top" : "");
    d.innerHTML = `<span>layer ${i}</span><span class="bar"><span style="--p:${p}"></span></span><span class="tok">"${tok.replace(/ /g, "␣")}"</span>`;
    layersEl.appendChild(d);
  });
  function animateLayers() {
    $$(".layer-row .bar span").forEach((s, i) => {
      s.style.transform = "scaleX(0)";
      setTimeout(() => (s.style.transform = `scaleX(${Math.min(1, parseFloat(s.style.getPropertyValue("--p")) * 4)})`), 60 + i * 70);
    });
  }
})();
