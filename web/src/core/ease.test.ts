import { describe, expect, it } from 'vitest';
import {
  accelerateIn,
  clamp01,
  cubicBezier,
  emphasized,
  emphasizedOut,
  expoOut,
  expoOutSoft,
  linear,
  overshoot,
  progress,
  smooth,
  smootherstep,
  smoothstep,
  spring,
} from './ease';

const NAMED = {
  linear,
  expoOut,
  expoOutSoft,
  emphasizedOut,
  emphasized,
  accelerateIn,
  smooth,
  overshoot,
  smoothstep,
  smootherstep,
  spring: spring(),
};

describe('ease', () => {
  it('linear is the identity on [0, 1]', () => {
    for (const x of [0, 0.25, 0.5, 0.73, 1]) {
      expect(linear(x)).toBe(x);
    }
  });

  it('a 1:1 cubic bezier is the identity', () => {
    const id = cubicBezier(0, 0, 1, 1);
    for (const x of [0, 0.1, 0.33, 0.5, 0.8, 1]) {
      expect(id(x)).toBeCloseTo(x, 5);
    }
  });

  it('every named curve starts at 0 and finishes at 1', () => {
    for (const [name, fn] of Object.entries(NAMED)) {
      expect(fn(0), name).toBeCloseTo(0, 5);
      expect(fn(1), name).toBeCloseTo(1, 5);
    }
  });

  it('clamps and treats a zero-duration window as a step', () => {
    expect(clamp01(-2)).toBe(0);
    expect(clamp01(2)).toBe(1);
    expect(progress(0.4, 1, 2)).toBe(0);
    expect(progress(2, 1, 2)).toBe(0.5);
    expect(progress(4, 1, 2)).toBe(1);
    expect(progress(0.5, 1, 0)).toBe(0);
    expect(progress(1, 1, 0)).toBe(1);
  });
});
