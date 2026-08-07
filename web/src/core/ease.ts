/**
 * Easing curves.
 *
 * The set is deliberately small. The discipline that separates expensive-feeling motion
 * from amateur motion is that most things use one boring workhorse ease-out, and only one
 * element per scene is allowed to be dramatic.
 */

export type EaseFn = (t: number) => number;

export const linear: EaseFn = (t) => t;

/** Cubic bezier solved by Newton then bisection. Accurate to ~1e-6 over [0,1]. */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number): EaseFn {
  const A = (a: number, b: number) => 1 - 3 * b + 3 * a;
  const B = (a: number, b: number) => 3 * b - 6 * a;
  const C = (a: number) => 3 * a;
  const calc = (t: number, a: number, b: number) => ((A(a, b) * t + B(a, b)) * t + C(a)) * t;
  const slope = (t: number, a: number, b: number) =>
    3 * A(a, b) * t * t + 2 * B(a, b) * t + C(a);

  if (x1 === y1 && x2 === y2) return linear;

  return (x: number) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;

    let t = x;
    for (let i = 0; i < 8; i++) {
      const d = slope(t, x1, x2);
      if (Math.abs(d) < 1e-6) break;
      const err = calc(t, x1, x2) - x;
      if (Math.abs(err) < 1e-6) return calc(t, y1, y2);
      t -= err / d;
    }
    let lo = 0;
    let hi = 1;
    t = x;
    for (let i = 0; i < 20; i++) {
      const err = calc(t, x1, x2) - x;
      if (Math.abs(err) < 1e-6) break;
      if (err > 0) hi = t;
      else lo = t;
      t = (lo + hi) / 2;
    }
    return calc(t, y1, y2);
  };
}

/**
 * Reaches ~90% of the distance in ~30% of the duration, then glides. This is the single
 * curve most responsible for motion reading as premium rather than mechanical, and it is
 * the default for anything entering frame.
 */
export const expoOut = cubicBezier(0.16, 1, 0.3, 1);

/** Softer expo, better for large or heavy objects. */
export const expoOutSoft = cubicBezier(0.22, 1, 0.36, 1);

/** Material 3 "emphasized decelerate": arrives and settles. */
export const emphasizedOut = cubicBezier(0.05, 0.7, 0.1, 1);

/** Material 3 "emphasized": symmetric, for a move between two resting places. */
export const emphasized = cubicBezier(0.2, 0, 0, 1);

/** For exits. Things should leave faster than they arrive. */
export const accelerateIn = cubicBezier(0.3, 0, 1, 1);

/** Gentle S-curve. The workhorse; use this when in doubt. */
export const smooth = cubicBezier(0.4, 0, 0.2, 1);

/**
 * Overshoots before settling. Strictly one element per scene. Never on the winning
 * digit: certainty should read as a lock, not a wobble.
 */
export const overshoot = cubicBezier(0.34, 1.56, 0.64, 1);

/** Classic smoothstep, useful inside shaders' JS-side equivalents. */
export const smoothstep: EaseFn = (t) => {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
};

/** Quintic smootherstep: zero first *and* second derivative at both ends. */
export const smootherstep: EaseFn = (t) => {
  const x = Math.min(1, Math.max(0, t));
  return x * x * x * (x * (x * 6 - 15) + 10);
};

/**
 * Critically-damped spring sampled analytically, so it stays a pure function of time and
 * remains seek-safe. `stiffness` in the react-spring sense; 210/20 is their "stiff"
 * preset and is what the final classification lock uses.
 */
export function spring(stiffness = 210, damping = 29, mass = 1): EaseFn {
  const w0 = Math.sqrt(stiffness / mass);
  const zeta = damping / (2 * Math.sqrt(stiffness * mass));
  if (zeta < 1) {
    const wd = w0 * Math.sqrt(1 - zeta * zeta);
    return (t) => {
      if (t >= 1) return 1;
      const x = t * 4; // map [0,1] onto ~4 natural time units so it settles by t=1
      return 1 - Math.exp(-zeta * w0 * x) * (Math.cos(wd * x) + ((zeta * w0) / wd) * Math.sin(wd * x));
    };
  }
  return (t) => {
    if (t >= 1) return 1;
    const x = t * 4;
    return 1 - (1 + w0 * x) * Math.exp(-w0 * x);
  };
}

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const clamp01 = (v: number) => clamp(v, 0, 1);

/** Normalised progress of `t` through the window [start, start+dur], clamped. */
export const progress = (t: number, start: number, dur: number) =>
  dur <= 0 ? (t >= start ? 1 : 0) : clamp01((t - start) / dur);
