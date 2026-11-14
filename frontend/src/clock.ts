/** Shared playback clock, mutated outside React so the WebGL layer can read it
 *  every frame without re-rendering the whole app. */
type Listener = () => void;

export const clock = {
  t: 0,
  duration: 0,
  playing: false,
  rate: 1,
  listeners: new Set<Listener>(),
  set(t: number) {
    this.t = Math.max(0, Math.min(this.duration, t));
    this.emit();
  },
  play(from?: number) {
    if (from !== undefined) this.t = from;
    if (this.t >= this.duration - 1e-3) this.t = 0;
    this.playing = true;
    this.emit();
  },
  pause() { this.playing = false; this.emit(); },
  emit() { this.listeners.forEach((l) => l()); },
  subscribe(l: Listener) { this.listeners.add(l); return () => { this.listeners.delete(l); }; },
};

let last = performance.now();
function tick(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (clock.playing) {
    clock.t += dt * clock.rate;
    if (clock.t >= clock.duration) { clock.t = clock.duration; clock.playing = false; }
    clock.emit();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
