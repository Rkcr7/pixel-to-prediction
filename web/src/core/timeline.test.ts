import { describe, expect, it, vi } from 'vitest';
import { Player, Timeline, TimelineBuilder } from './timeline';

describe('Timeline', () => {
  it('holds from before the first clip and to after the last', () => {
    const tl = new Timeline([{ key: 'a', at: 1, dur: 2, from: 10, to: 20, ease: (t) => t }]);
    expect(tl.sample(0).a).toBe(10);
    expect(tl.sample(1).a).toBe(10);
    expect(tl.sample(2).a).toBe(15);
    expect(tl.sample(3).a).toBe(20);
    expect(tl.sample(9).a).toBe(20);
  });

  it('treats a zero duration as a step', () => {
    const tl = new Timeline([{ key: 'a', at: 1, dur: 0, from: 0, to: 1 }]);
    expect(tl.sample(0.999).a).toBe(0);
    expect(tl.sample(1).a).toBe(1);
  });

  it('is a pure function of time: forward and reverse scans agree', () => {
    const tl = new TimelineBuilder()
      .then('x', 2, 0, 1)
      .then('y', 1, 4, 0)
      .build();
    const times = [0, 0.25, 0.8, 1.5, 2, 2.5, 3, 4];
    const forward = times.map((t) => ({ ...tl.sample(t) }));
    const reverse = [...times].reverse().map((t) => ({ ...tl.sample(t) }));
    expect(reverse.reverse()).toEqual(forward);
  });

  it('reports cues in (from, to]', () => {
    const tl = new Timeline([], [
      { name: 'a', at: 1 },
      { name: 'b', at: 2 },
      { name: 'c', at: 3 },
    ]);
    expect(tl.cuesBetween(1, 2).map((c) => c.name)).toEqual(['b']);
    expect(tl.cuesBetween(2, 2)).toEqual([]);
    expect(tl.cuesBetween(3, 1)).toEqual([]);
  });

  it('staggers by rank, not by index', () => {
    const tl = new TimelineBuilder()
      .stagger((i) => `m${i}`, 3, 1, 0, 1, 0.5, undefined, [2, 0, 1])
      .build();
    // order [2, 0, 1] means index 2 is first, then 0, then 1.
    expect(tl.sample(0.01).m2).toBeGreaterThan(0);
    expect(tl.sample(0.01).m0).toBe(0);
    expect(tl.sample(0.51).m0).toBeGreaterThan(0);
    expect(tl.sample(0.51).m1).toBe(0);
  });
});

describe('Player', () => {
  it('does not fire cues on seek', () => {
    const tl = new Timeline([{ key: 'a', at: 0, dur: 2, from: 0, to: 1 }], [{ name: 'hit', at: 1 }]);
    const player = new Player();
    const onCue = vi.fn();
    player.onCue = onCue;
    player.load(tl, false);
    player.seek(2);
    expect(onCue).not.toHaveBeenCalled();
    expect(player.time).toBe(2);
  });

  it('fires each cue once while advancing forward', () => {
    const tl = new Timeline([{ key: 'a', at: 0, dur: 2, from: 0, to: 1 }], [{ name: 'hit', at: 1 }]);
    const player = new Player();
    const onCue = vi.fn();
    player.onCue = onCue;
    player.load(tl, true);
    player.advance(0.6);
    expect(onCue).not.toHaveBeenCalled();
    player.advance(0.6);
    expect(onCue).toHaveBeenCalledTimes(1);
    expect(onCue.mock.calls[0][0].name).toBe('hit');
    player.advance(0.6);
    expect(onCue).toHaveBeenCalledTimes(1);
  });
});
