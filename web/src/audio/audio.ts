/**
 * Procedural sound. No audio files: everything is synthesised, so the whole soundtrack
 * costs zero bytes of transfer and stays in sync with a timeline that can be scrubbed
 * and re-timed.
 *
 * Tuned to a minor pentatonic so that any two cues landing together still sound
 * intentional. The mix sits low on purpose; this should register as atmosphere rather
 * than as a soundtrack, and it must not be the reason someone mutes the tab.
 */

const PENTATONIC = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33];

export type CueName =
  | 'enter'
  | 'tick'
  | 'whoosh'
  | 'settle'
  | 'sweep'
  | 'bloom'
  | 'reveal'
  | 'cut'
  | 'contest'
  | 'contract'
  | 'flow'
  | 'stretch'
  | 'pour'
  | 'lock'
  | 'arrive';

export class Audio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private padGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private started = false;
  private step = 0;

  enabled: boolean;

  constructor() {
    // localStorage throws in a sandboxed iframe and under some privacy settings. This
    // runs during module evaluation, outside the app's own try/catch, so an unguarded
    // access here leaves the loading screen up forever with no error shown.
    let stored: string | null = null;
    try {
      stored = localStorage.getItem('p2p.sound');
    } catch {
      /* storage unavailable; fall back to the default */
    }
    this.enabled = stored !== 'off';
  }

  setEnabled(on: boolean) {
    this.enabled = on;
    try {
      localStorage.setItem('p2p.sound', on ? 'on' : 'off');
    } catch {
      /* preference just will not persist */
    }
    if (this.master && this.ctx) {
      this.master.gain.cancelScheduledValues(this.ctx.currentTime);
      this.master.gain.setTargetAtTime(on ? 0.5 : 0, this.ctx.currentTime, 0.08);
    }
    if (on) void this.ensure();
  }

  /** Must be called from a user gesture. Safe to call repeatedly. */
  async ensure(): Promise<void> {
    if (!this.ctx) {
      const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return;
      this.ctx = new Ctor();
      this.master = this.ctx.createGain();
      this.master.gain.value = this.enabled ? 0.5 : 0;
      this.master.connect(this.ctx.destination);
      this.buildNoise();
    }
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  private buildNoise() {
    const ctx = this.ctx;
    if (!ctx) return;
    const len = Math.floor(ctx.sampleRate * 1.5);
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    this.noiseBuffer = buf;
  }

  /** A quiet two-oscillator drone under the whole run. */
  startPad() {
    const ctx = this.ctx;
    if (!ctx || !this.master || this.started) return;
    this.started = true;

    const pad = ctx.createGain();
    pad.gain.value = 0;
    pad.connect(this.master);
    this.padGain = pad;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 620;
    filter.Q.value = 0.6;
    filter.connect(pad);

    for (const [freq, detune] of [
      [110, -4],
      [110, 5],
      [164.81, 2],
    ] as const) {
      const osc = ctx.createOscillator();
      osc.type = 'sawtooth';
      osc.frequency.value = freq;
      osc.detune.value = detune;
      const g = ctx.createGain();
      g.gain.value = 0.06;
      osc.connect(g).connect(filter);
      osc.start();

      // Slow drift so the drone never sits perfectly still.
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.05 + Math.random() * 0.06;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = 3;
      lfo.connect(lfoGain).connect(osc.detune);
      lfo.start();
    }

    pad.gain.setTargetAtTime(0.34, ctx.currentTime, 1.4);
  }

  private capture: MediaStreamAudioDestinationNode | null = null;

  /** A stream of the same mix, so a recorded clip carries its own soundtrack. */
  captureStream(): MediaStream | null {
    if (!this.ctx || !this.master || !this.enabled) return null;
    this.releaseStream();
    this.capture = this.ctx.createMediaStreamDestination();
    this.master.connect(this.capture);
    return this.capture.stream;
  }

  /** Detach the recording tap. Without this each recording leaves the master gain
   *  permanently fanned out to another destination node. */
  releaseStream() {
    if (!this.capture || !this.master) return;
    try {
      this.master.disconnect(this.capture);
    } catch {
      /* already disconnected */
    }
    this.capture = null;
  }

  fadePad(target: number, seconds = 0.8) {
    if (!this.ctx || !this.padGain) return;
    this.padGain.gain.setTargetAtTime(target, this.ctx.currentTime, seconds / 3);
  }

  private tone(freq: number, duration: number, gain: number, type: OscillatorType = 'sine', detune = 0) {
    const ctx = this.ctx;
    if (!ctx || !this.master || !this.enabled) return;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = freq;
    osc.detune.value = detune;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + duration);
    osc.connect(g).connect(this.master);
    osc.start(t);
    osc.stop(t + duration + 0.05);
  }

  private noise(duration: number, gain: number, from: number, to: number, q = 1) {
    const ctx = this.ctx;
    if (!ctx || !this.master || !this.noiseBuffer || !this.enabled) return;
    const t = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuffer;
    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.Q.value = q;
    filter.frequency.setValueAtTime(from, t);
    filter.frequency.exponentialRampToValueAtTime(Math.max(40, to), t + duration);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + duration * 0.18);
    g.gain.exponentialRampToValueAtTime(0.0001, t + duration);
    src.connect(filter).connect(g).connect(this.master);
    src.start(t);
    src.stop(t + duration + 0.05);
  }

  play(cue: CueName) {
    if (!this.enabled || !this.ctx) return;
    switch (cue) {
      case 'enter':
        this.tone(PENTATONIC[0], 1.6, 0.1, 'sine');
        this.tone(PENTATONIC[3], 1.2, 0.05, 'sine');
        break;
      case 'tick':
        this.tone(PENTATONIC[4 + (this.step++ % 3)], 0.28, 0.055, 'triangle');
        break;
      case 'whoosh':
        this.noise(0.55, 0.05, 340, 1900, 0.8);
        break;
      case 'settle':
        this.tone(PENTATONIC[2], 0.7, 0.06, 'sine');
        this.tone(PENTATONIC[5], 0.5, 0.03, 'sine');
        break;
      case 'sweep':
        this.noise(1.5, 0.038, 260, 2600, 1.6);
        this.tone(PENTATONIC[1], 1.4, 0.035, 'sine');
        break;
      case 'bloom':
        this.tone(PENTATONIC[4], 0.9, 0.06, 'sine');
        this.tone(PENTATONIC[6], 0.8, 0.04, 'sine', 4);
        break;
      case 'reveal':
        this.tone(PENTATONIC[3], 0.6, 0.05, 'triangle');
        break;
      case 'cut':
        // The one moment allowed to feel abrupt: signal being destroyed.
        this.tone(70, 0.42, 0.16, 'sine');
        this.noise(0.3, 0.055, 1500, 180, 0.7);
        break;
      case 'contest':
        this.tone(PENTATONIC[5], 0.16, 0.045, 'square');
        break;
      case 'contract':
        this.tone(PENTATONIC[2], 0.5, 0.07, 'sine');
        this.noise(0.35, 0.03, 900, 240, 1.2);
        break;
      case 'flow':
        this.noise(1.1, 0.026, 500, 1400, 2.2);
        break;
      case 'stretch':
        this.tone(PENTATONIC[1], 0.75, 0.06, 'sawtooth', -6);
        this.tone(PENTATONIC[4], 0.75, 0.035, 'sine', 6);
        break;
      case 'pour':
        this.noise(0.85, 0.032, 1600, 500, 1.4);
        this.tone(PENTATONIC[3], 0.7, 0.05, 'sine');
        break;
      case 'lock':
        // The resolution. A clean triad, and the only cue with any weight to it.
        this.tone(PENTATONIC[0], 1.9, 0.1, 'sine');
        this.tone(PENTATONIC[3], 1.7, 0.075, 'sine');
        this.tone(PENTATONIC[5], 1.5, 0.055, 'sine');
        this.tone(PENTATONIC[7], 1.3, 0.03, 'sine');
        break;
      case 'arrive':
        this.tone(PENTATONIC[6], 1.1, 0.05, 'sine');
        break;
    }
  }
}
