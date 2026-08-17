import { describe, expect, it } from 'vitest';
import type { Evidence } from '../core/engine';
import { explain, formatConfidence, verdict } from './copy';

const base = (over: Partial<Evidence> = {}): Evidence => ({
  top1: 3,
  top2: 8,
  p1: 0.94,
  p2: 0.04,
  margin: 4,
  holes: 0,
  inkRatio: 0.12,
  regions: [0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.1, 0.05],
  counterRegions: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  strongestRegion: 4,
  contestedRegion: 7,
  diffuse: false,
  ...over,
});

describe('explain', () => {
  it('hedges only when confidence is actually low', () => {
    expect(explain(base({ p1: 0.94 })).hedged).toBe(false);
    expect(explain(base({ p1: 0.69 })).hedged).toBe(true);
  });

  it('does not invent a single-region cause when the evidence is diffuse', () => {
    const words = explain(base({ diffuse: true, holes: 0 }));
    expect(words.reason).toMatch(/spread/i);
    expect(words.reason).not.toMatch(/leaned on the ink/i);
  });

  it('is deterministic: the same evidence always produces the same words', () => {
    const e = base({ p1: 0.81, holes: 1, strongestRegion: 2 });
    expect(explain(e)).toEqual(explain(e));
    expect(verdict(e)).toBe(verdict(e));
    expect(formatConfidence(e.p1)).toBe(formatConfidence(e.p1));
  });
});

describe('verdict', () => {
  it('softens the claim as confidence falls', () => {
    expect(verdict(base({ p1: 0.999 }))).toBe('That is a 3.');
    expect(verdict(base({ p1: 0.95 }))).toBe('It says 3.');
    expect(verdict(base({ p1: 0.7 }))).toBe('It thinks 3.');
    expect(verdict(base({ p1: 0.4 }))).toBe('Its best guess is 3.');
  });
});

describe('formatConfidence', () => {
  it('does not claim 100% and keeps one decimal only in the high nineties', () => {
    expect(formatConfidence(1)).toBe('>99.9%');
    expect(formatConfidence(0.996)).toBe('99.6%');
    expect(formatConfidence(0.81)).toBe('81%');
  });
});
