/**
 * A deterministic, seek-safe timeline.
 *
 * Every animated value is recomputed from scratch on each sample. Nothing accumulates and
 * nothing depends on the previous frame, so scrubbing backward gives bit-identical results
 * to playing forward, speed changes are free, and an offline render matches what the user
 * saw on screen exactly.
 */

import { clamp01, expoOut, type EaseFn } from './ease';

export interface Clip {
  /** Name of the value this clip drives. */
  key: string;
  /** Absolute start time in seconds. */
  at: number;
  /** Duration in seconds. Zero makes it a step change at `at`. */
  dur: number;
  from: number;
  to: number;
  ease?: EaseFn;
}

/** A one-shot marker. Used for sound and for stage-entered callbacks. */
export interface Cue {
  name: string;
  at: number;
  /** Free-form payload passed through to the listener. */
  data?: unknown;
}

export type Sampled = Record<string, number>;

export class Timeline {
  readonly clips: readonly Clip[];
  readonly cues: readonly Cue[];
  readonly duration: number;
  /** Clips grouped by key and sorted by start time, so sampling is a short scan. */
  private readonly byKey: Map<string, Clip[]>;
  private readonly defaults: Sampled;

  constructor(clips: Clip[], cues: Cue[] = [], minDuration = 0) {
    this.clips = clips;
    this.cues = [...cues].sort((a, b) => a.at - b.at);

    const byKey = new Map<string, Clip[]>();
    for (const c of clips) {
      const list = byKey.get(c.key);
      if (list) list.push(c);
      else byKey.set(c.key, [c]);
    }
    for (const list of byKey.values()) list.sort((a, b) => a.at - b.at);
    this.byKey = byKey;

    // Before its first clip starts, a key holds that clip's `from` value.
    const defaults: Sampled = {};
    for (const [key, list] of byKey) defaults[key] = list[0].from;
    this.defaults = defaults;

    this.duration = Math.max(
      minDuration,
      ...clips.map((c) => c.at + c.dur),
      ...cues.map((c) => c.at),
      0,
    );
  }

  /** Keys this timeline drives. */
  keys(): string[] {
    return [...this.byKey.keys()];
  }

  /**
   * Evaluate every key at time `t`. Writes into `out` when supplied so the render loop
   * can reuse one object and allocate nothing per frame.
   */
  sample(t: number, out: Sampled = {}): Sampled {
    for (const [key, list] of this.byKey) {
      let value = this.defaults[key];
      for (let i = 0; i < list.length; i++) {
        const c = list[i];
        if (t < c.at) break;
        if (c.dur <= 0 || t >= c.at + c.dur) {
          value = c.to;
          continue;
        }
        const p = clamp01((t - c.at) / c.dur);
        const e = (c.ease ?? expoOut)(p);
        value = c.from + (c.to - c.from) * e;
        break;
      }
      out[key] = value;
    }
    return out;
  }

  /** Cues strictly inside (from, to]. Used to fire sound only when moving forward. */
  cuesBetween(from: number, to: number): Cue[] {
    if (to <= from) return [];
    return this.cues.filter((c) => c.at > from && c.at <= to);
  }
}

/**
 * Small helper for writing timelines as a readable script rather than a pile of absolute
 * timestamps. `at()` moves a cursor; `add()` places clips relative to it.
 */
export class TimelineBuilder {
  private clips: Clip[] = [];
  private cues: Cue[] = [];
  private cursor = 0;

  /** Move the cursor to an absolute time. */
  at(t: number): this {
    this.cursor = t;
    return this;
  }

  /** Advance the cursor. Negative values overlap the previous beat. */
  wait(dt: number): this {
    this.cursor += dt;
    return this;
  }

  get now(): number {
    return this.cursor;
  }

  /** Add a clip starting at the cursor. Does not move the cursor. */
  add(key: string, dur: number, from: number, to: number, ease?: EaseFn): this {
    this.clips.push({ key, at: this.cursor, dur, from, to, ease });
    return this;
  }

  /** Add a clip and advance the cursor past it. */
  then(key: string, dur: number, from: number, to: number, ease?: EaseFn): this {
    this.add(key, dur, from, to, ease);
    this.cursor += dur;
    return this;
  }

  /**
   * Stagger the same transition across `n` elements. `order` maps element index to its
   * place in the queue, which lets callers stagger by activation magnitude rather than
   * by index: the strongest feature lights first, so the ordering itself carries meaning.
   */
  stagger(
    keyOf: (i: number) => string,
    n: number,
    dur: number,
    from: number,
    to: number,
    step: number,
    ease?: EaseFn,
    order?: number[],
  ): this {
    const base = this.cursor;
    for (let i = 0; i < n; i++) {
      const slot = order ? order.indexOf(i) : i;
      this.clips.push({
        key: keyOf(i),
        at: base + (slot < 0 ? i : slot) * step,
        dur,
        from,
        to,
        ease,
      });
    }
    return this;
  }

  cue(name: string, data?: unknown): this {
    this.cues.push({ name, at: this.cursor, data });
    return this;
  }

  build(minDuration = 0): Timeline {
    return new Timeline(this.clips, this.cues, minDuration);
  }
}

export type PlayState = 'idle' | 'playing' | 'paused' | 'finished';

/**
 * Drives a timeline from real time. The player owns exactly one number, `time`; every
 * other piece of visual state is derived from it.
 */
export class Player {
  time = 0;
  speed = 1;
  state: PlayState = 'idle';
  timeline: Timeline | null = null;

  onCue: ((cue: Cue) => void) | null = null;
  onStateChange: ((state: PlayState) => void) | null = null;

  load(timeline: Timeline, autoplay = true) {
    this.timeline = timeline;
    this.time = 0;
    this.setState(autoplay ? 'playing' : 'paused');
  }

  private setState(s: PlayState) {
    if (this.state === s) return;
    this.state = s;
    this.onStateChange?.(s);
  }

  play() {
    if (!this.timeline) return;
    if (this.state === 'finished') this.time = 0;
    this.setState('playing');
  }

  pause() {
    if (this.state === 'playing') this.setState('paused');
  }

  toggle() {
    if (this.state === 'playing') this.pause();
    else this.play();
  }

  /** Jump to an absolute time. Never fires cues: seeking is silent by design. */
  seek(t: number) {
    if (!this.timeline) return;
    this.time = Math.min(this.timeline.duration, Math.max(0, t));
    if (this.state === 'finished' && this.time < this.timeline.duration) {
      this.setState('paused');
    }
  }

  restart() {
    this.time = 0;
    this.setState('playing');
  }

  /** Advance by real elapsed seconds. Returns the current time. */
  advance(dt: number): number {
    if (!this.timeline || this.state !== 'playing') return this.time;
    const prev = this.time;
    const next = Math.min(this.timeline.duration, prev + dt * this.speed);
    this.time = next;

    if (this.onCue) {
      for (const cue of this.timeline.cuesBetween(prev, next)) this.onCue(cue);
    }
    if (next >= this.timeline.duration) this.setState('finished');
    return next;
  }

  get progress(): number {
    if (!this.timeline || this.timeline.duration <= 0) return 0;
    return this.time / this.timeline.duration;
  }
}
