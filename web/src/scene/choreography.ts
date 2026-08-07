/**
 * The score.
 *
 * Each operation is a four-part beat: anticipation (the target draws in, signalling
 * "watch here"), action (the transformation, and nothing else in frame moves), settle
 * (overshoot resolves, glow decays), hold (nothing moves at all — this is where
 * comprehension actually happens).
 *
 * The camera moves only between operations and is locked during them.
 *
 * Two rules govern the copy, both from CNN Explainer's user study, which is the only
 * thing in this field that was tested on real people rather than asserted by a designer.
 * Their top complaint was how-without-why ("I know how to compute them, but I don't know
 * why we compute them"), and participants reported understanding convolution only after
 * reading the annotation text, not from the animation. So: every stage answers a why,
 * and a note gets `words / 2.5` seconds on screen with nothing else moving.
 */

import { accelerateIn, emphasized, emphasizedOut, expoOut, smooth, spring } from '../core/ease';
import { Timeline, TimelineBuilder } from '../core/timeline';
import type { Run } from '../core/engine';
import { C1, C2, CLASSES, HIDDEN } from './constants';

export interface StageInfo {
  index: number;
  key: string;
  title: string;
  /** Plain-language line shown while this stage plays. */
  caption: string;
  start: number;
  end: number;
}

export interface Score {
  timeline: Timeline;
  stages: StageInfo[];
}

// Critically damped on purpose. The winner locking is the one moment that must not
// wobble, and 210/20 is underdamped (zeta about 0.69) with roughly 5% overshoot.
const lock = spring(210, 29);

/**
 * A claim, held long enough to actually be read.
 *
 * A first-time reader parses explanatory prose at roughly 2.5 words per second, well
 * below silent-reading speed. Anything shorter than that is decoration that happens to
 * contain words.
 */
function note(b: TimelineBuilder, key: string, at: number, text: string) {
  const words = text.trim().split(/\s+/).length;
  const hold = Math.max(3.4, words / 2.5);
  b.at(at).add(key, 0.4, 0, 1, smooth);
  b.at(at + 0.4 + hold).add(key, 0.45, 1, 0, smooth);
  return hold;
}

/** Every note in the piece, so the copy can be read and edited in one place. */
export const NOTES: Record<string, string> = {
  'n.pixels':
    'Your drawing is a big grid of pixels. The network never sees it at that size.',
  'n.numbers':
    'To the network this is not a picture. It is 784 numbers, one per square, running from 0 for empty to 1 for solid ink.',
  'n.share':
    'The filter never changed as it moved. The same 25 numbers, used at every position.',
  'n.relu':
    'Every negative response was set to zero and thrown away. Every positive one was left exactly as it was.',
  'n.pool':
    'Each 2 by 2 square keeps only its brightest cell, so the picture halves. What matters is that an edge is there, not exactly where.',
  'n.flatten':
    'Those 16 small maps are now just a list of 784 numbers. The same count it started with, but each one means a feature rather than a dot of ink.',
  'n.template':
    'Every hidden unit carries its own picture of what it hopes to find. Compare that against what your digit actually has.',
  'n.agree':
    'Multiply the two together, square by square. Whatever is left is the evidence this unit found.',
  'n.gate':
    'Add it all up, add this unit’s own bias, and if the total lands above zero the unit fires.',
  'n.exp':
    'Raising each score to a power does more than make it positive. It blows the gaps wide open, so a small lead becomes a huge one.',
  'n.close':
    'It has never seen your handwriting before. It learned from sixty thousand other people’s.',
};

export function buildScore(run: Run): Score {
  const b = new TimelineBuilder();
  const stages: StageInfo[] = [];
  const openStage = (key: string, title: string, caption: string) => {
    stages.push({ index: stages.length, key, title, caption, start: b.now, end: 0 });
  };
  const closeStage = () => {
    stages[stages.length - 1].end = b.now;
  };

  // Global look. Aberration starts high and resolves to zero as the network commits:
  // while it is undecided the image is literally unresolved.
  b.at(0).add('post.fade', 0.7, 0, 1, smooth);
  b.at(0).add('post.aberration', 1.2, 0.018, 0.006, smooth);
  b.at(0).add('post.groundGlow', 1.4, 0, 0.34, smooth);
  b.at(0).add('post.exposure', 1.0, 0.75, 1.0, smooth);
  b.at(0).add('post.defocus', 0.9, 0.35, 0, expoOut);

  // ---------------------------------------------------------------- Stage 1
  openStage('input', 'Your digit', 'This is exactly what you handed the network.');

  b.at(0.15).add('s1.digit', 0.95, 0, 1, expoOut);
  b.at(0.15).add('s1.scale', 1.1, 0.84, 1, emphasizedOut);
  b.at(0.3).cue('enter');
  note(b, 'n.pixels', 0.9, NOTES['n.pixels']);
  b.at(2.6);
  closeStage();

  // ---------------------------------------------------------------- Stage 2
  // BUT: it cannot use that directly.
  openStage(
    'prepare',
    'Preparing the input',
    'It cannot use that directly. First the drawing gets cropped, shrunk into a 20-pixel box and re-centred on its centre of mass, exactly the way MNIST was built.',
  );

  b.at(2.7).add('s2.cropRing', 0.5, 0, 1, expoOut).cue('tick');
  b.at(2.7).add('s1.digit', 0.4, 1, 0.55, smooth);

  b.at(3.6).add('s2.boxed', 0.75, 0, 1, emphasized).cue('whoosh');
  b.at(3.6).add('s2.source', 0.6, 1, 0, smooth);
  b.at(3.6).add('s2.cropRing', 0.5, 1, 0, accelerateIn);

  b.at(4.7).add('s2.comDot', 0.4, 0, 1, expoOut).cue('tick');
  b.at(5.15).add('s2.centre', 0.7, 0, 1, emphasized);
  b.at(5.6).add('s2.comDot', 0.5, 1, 0, smooth);

  b.at(5.85).add('s2.input', 0.65, 0, 1, emphasizedOut);
  b.at(5.85).add('s2.boxed', 0.5, 1, 0, smooth);
  b.at(6.05).add('s2.grid', 0.7, 0, 1, smooth).cue('settle');
  note(b, 'n.numbers', 6.3, NOTES['n.numbers']);
  b.at(7.2);
  closeStage();

  // ---------------------------------------------------------------- Stage 3
  // THEREFORE: now it can look.
  openStage(
    'edges',
    'First look',
    'Now it can look. Eight small filters sweep across the image, and each one lights up wherever it finds the edge it was trained to spot.',
  );

  // Camera travels between operations, never during one.
  b.at(7.2).add('cam.station', 1.3, 0, 1, emphasized);
  // Straight to zero, not merely dim: the camera passes z=0 on the way, and anything
  // still alive there ends up between the lens and the subject.
  b.at(7.2).add('s2.input', 0.5, 1, 0, smooth);
  b.at(7.2).add('s2.grid', 0.5, 1, 0, smooth);

  // The kernels themselves: the actual trained weights, not an illustration.
  b.at(7.9).stagger((i) => `s3.kernel${i}`, C1, 0.45, 0, 1, 0.06, expoOut);
  b.at(8.5).cue('tick');

  // One hero filter reads across the plate at full fidelity. Showing all eight sweeping
  // at once is unreadable; the rest resolve afterwards as an already-finished grid.
  b.at(9.3).add('s3.hero', 0.5, 0, 1, expoOut);
  // Push in for the sweep, and bring the aim down with it: the station centre is set
  // high to include the filter row, which is no longer the subject once we are close.
  b.at(9.3).add('cam.zoom', 1.0, 1, 0.78, emphasized);
  b.at(9.3).add('cam.centerAdjust', 1.0, 0, -0.55, emphasized);
  // The working filter comes down out of the row to sit beside its own output.
  b.at(9.3).add('s3.heroLift', 0.75, 0, 1, emphasized);

  b.at(9.5).add('s3.sweep', 2.4, 0, 1, smooth).cue('sweep');
  b.at(9.5).add('s3.heroKernel', 2.4, 0, 1, smooth);
  // Negative responses appear as they are computed. Revealing them later would imply
  // ReLU had already run, which is the wrong order.
  b.at(9.55).add('s3.signed', 0.5, 0, 1, smooth);

  note(b, 'n.share', 12.1, NOTES['n.share']);

  // The other seven, staggered by how strongly each fired rather than by index, so the
  // order of appearance is itself information.
  b.at(12.8).stagger((i) => `s3.plate${i}`, C1, 0.5, 0, 1, 0.07, expoOut, run.conv1Order);
  b.at(12.8).add('s3.spread', 0.95, 0, 1, emphasizedOut).cue('bloom');
  b.at(12.8).add('cam.zoom', 1.0, 0.78, 1, emphasized);
  b.at(12.8).add('cam.centerAdjust', 1.0, -0.55, 0, emphasized);
  b.at(12.8).add('s3.heroLift', 0.7, 1, 0, smooth);

  // ReLU. Anticipation first: the negative half brightens so the eye is already on it
  // when it goes. Then the cut. The positive half is untouched throughout — that
  // asymmetry is the whole lesson, so nothing about it may change here.
  b.at(15.4).add('s3.reluHint', 0.5, 0, 1, smooth).cue('reveal');
  b.at(16.4).add('s3.relu', 0.75, 0, 1, emphasized).cue('cut');
  b.at(16.4).add('s3.reluHint', 0.6, 1, 0, smooth);
  note(b, 'n.relu', 17.2, NOTES['n.relu']);
  b.at(19.5);
  closeStage();

  // ---------------------------------------------------------------- Stage 4
  // BUT: one filter can only ever find one thing.
  openStage(
    'shapes',
    'Finding shapes',
    'But one filter can only ever find one thing. So neighbouring cells compete, the picture halves, and the survivors get combined into richer features.',
  );

  // Come back to one map and look closely. At grid scale a 2x2 window is two pixels
  // wide, so pooling is literally invisible unless we zoom in for it.
  b.at(19.5).add('s4.focus', 0.9, 0, 1, emphasized);
  b.at(19.5).add('cam.zoom', 0.9, 1, 0.78, emphasized);
  b.at(19.5).add('cam.centerAdjust', 0.9, 0, -0.62, emphasized);

  b.at(21.0).add('s4.contest', 0.32, 0, 1, smooth).cue('contest');
  b.at(21.32).add('s4.contest', 0.28, 1, 0, smooth);
  b.at(21.4).add('s4.contract', 0.9, 0, 1, emphasized).cue('contract');
  note(b, 'n.pool', 22.4, NOTES['n.pool']);

  // Leave this station in the arrangement it is already in.
  //
  // Releasing s4.focus would send the hero plate back to its grid slot AND restore the
  // other seven from 4% opacity to full, right as the camera starts moving. Eight plates
  // popping back to full brightness while the lens rushes past them is the messiest
  // moment in the piece. Fade the whole thing out focused instead: there is no reason to
  // rebuild a layout nobody is going to look at again.
  b.at(23.4).add('s3.fade', 0.85, 1, 0, smooth);
  b.at(23.9).add('cam.station', 1.35, 1, 2, emphasized);
  b.at(23.9).add('cam.zoom', 0.9, 0.78, 1, smooth);
  b.at(23.9).add('cam.centerAdjust', 0.9, -0.62, 0, smooth);
  // Only now is it safe to reset, with the station already invisible.
  b.at(24.9).add('s4.focus', 0.3, 1, 0, smooth);

  // Only once the camera has actually arrived. Staggering these in mid-flight makes
  // them appear off-centre and then slide, which reads as a glitch rather than a reveal.
  b.at(25.3).stagger((i) => `s4.plate${i}`, C2, 0.45, 0, 1, 0.04, expoOut, run.conv2Order);
  b.at(25.3).cue('bloom');

  // Selective luminance: the maps that matter stay lit, the rest recede. The strongest
  // guidance device available, and it costs nothing.
  b.at(27.1).add('s4.rank', 0.75, 0, 1, smooth).cue('reveal');
  b.at(28.3).add('s4.pool', 0.7, 0, 1, emphasized).cue('contract');
  b.at(29.6);
  closeStage();

  // ---------------------------------------------------------------- Stage 5
  // THEREFORE: those features are now just numbers.
  openStage(
    'matching',
    'Matching possibilities',
    'Those features are now just 784 numbers. Every one of them votes, through 32 hidden units, on all ten digits at once.',
  );

  b.at(29.6).add('cam.station', 1.4, 2, 3, emphasized);
  b.at(29.6).add('s4.fade', 0.9, 1, 0, smooth);

  b.at(30.4).add('s5.block', 0.7, 0, 1, expoOut);
  note(b, 'n.flatten', 30.9, NOTES['n.flatten']);

  // The dense layer drawn as an actual dot product. Without this the 32 lit and unlit
  // sockets that follow are unexplained decoration: nothing tells a viewer what a hidden
  // unit computes, or why half of them stay dark.
  b.at(33.1).add('s5.panels', 1.0, 0, 1, emphasized).cue('whoosh');
  // Push in: three panels of detail framed for the whole dense station are too small to
  // read the cells that carry the entire point.
  b.at(33.1).add('cam.zoom', 1.0, 1, 0.82, emphasized);
  note(b, 'n.template', 33.7, NOTES['n.template']);

  b.at(36.1).add('s5.agree', 0.8, 0, 1, emphasizedOut).cue('reveal');
  note(b, 'n.agree', 36.5, NOTES['n.agree']);

  b.at(38.5).add('s5.sum', 0.95, 0, 1, emphasized).cue('pour');
  b.at(39.6).add('s5.gate', 0.8, 0, 1, emphasized).cue('tick');
  note(b, 'n.gate', 39.8, NOTES['n.gate']);

  // Now the lattice means something.
  b.at(42.7).add('s5.lattice', 0.9, 0, 1, smooth);
  b.at(42.7).add('cam.zoom', 0.9, 0.82, 1, smooth);
  b.at(42.7).stagger((i) => `s5.unit${i}`, HIDDEN, 0.4, 0, 1, 0.02, expoOut);

  b.at(44.1).add('s5.flowA', 0.7, 0, 1, smooth).cue('flow');
  b.at(44.7).stagger((i) => `s5.cand${i}`, CLASSES, 0.45, 0, 1, 0.05, expoOut, run.ranked);
  b.at(45.4).add('s5.flowB', 0.8, 0, 1, smooth).cue('flow');
  b.at(46.2).add('s5.weigh', 0.9, 0, 1, emphasizedOut).cue('reveal');
  b.at(48.1);
  closeStage();

  // ---------------------------------------------------------------- Stage 6
  // BUT: those ten scores do not add up to anything.
  openStage(
    'decision',
    'The decision',
    'But those ten scores do not add up to anything yet. Stretch the gaps with an exponential, then split one single unit of certainty between them.',
  );

  b.at(48.1).add('s6.gather', 1.0, 0, 1, emphasized).cue('whoosh');
  b.at(48.1).add('s5.flowA', 0.5, 1, 0, smooth);
  // All the way to zero: leftover particles drifting across the bars in stage 6 read as
  // noise, and this beat needs an uncluttered frame to be legible.
  b.at(48.1).add('s5.flowB', 0.6, 1, 0, smooth);

  // The raw logits. Some are negative and they sum to nothing in particular.
  b.at(49.1).add('s6.logits', 0.75, 0, 1, emphasizedOut).cue('tick');

  // Exponentiate: small gaps become large ones. Dramatic for free, and honest.
  b.at(50.5).add('s6.exp', 0.85, 0, 1, emphasized).cue('stretch');
  note(b, 'n.exp', 50.7, NOTES['n.exp']);

  // Normalise: one fixed unit of light divided ten ways. The container arrives on the
  // SAME clip, not before it: the landmark appearing exactly as the units change is what
  // makes the unit change legible.
  b.at(53.1).add('s6.budget', 0.9, 0, 1, emphasized);
  b.at(53.1).add('s6.normalize', 0.9, 0, 1, emphasized).cue('pour');

  // The winner locks. Losers may overshoot; the winner may not — certainty should read
  // as a lock, not a wobble.
  b.at(54.05).add('s6.lock', 0.75, 0, 1, lock).cue('lock');
  b.at(54.05).add('post.aberration', 0.8, 0.006, 0, smooth);
  b.at(55.0);
  closeStage();

  // ---------------------------------------------------------------- Stage 7
  openStage('answer', 'The answer', '');

  b.at(55.0).add('cam.station', 1.25, 3, 4, emphasized);
  b.at(55.0).add('s6.fade', 0.9, 1, 0, smooth);
  b.at(55.2).add('post.defocus', 0.7, 0.25, 0, expoOut);

  b.at(55.7).add('s7.digit', 0.85, 0, 1, emphasizedOut).cue('arrive');
  b.at(56.3).add('s7.saliency', 0.95, 0, 1, smooth).cue('reveal');
  b.at(57.3).add('s7.counter', 0.95, 0, 1, smooth);
  note(b, 'n.close', 57.7, NOTES['n.close']);
  b.at(58.6).add('s7.counter', 0.8, 1, 0.35, smooth);
  b.at(61.5);
  closeStage();

  return { timeline: b.build(61.5), stages };
}

/** Which stage contains time `t`. */
export function stageAt(stages: StageInfo[], t: number): StageInfo {
  for (let i = stages.length - 1; i >= 0; i--) {
    if (t >= stages[i].start) return stages[i];
  }
  return stages[0];
}
