/**
 * The score.
 *
 * Each operation is a four-part beat: anticipation (the target draws in, signalling
 * "watch here"), action (the transformation, and nothing else in frame moves), settle
 * (overshoot resolves, glow decays), hold (nothing moves at all — this is where
 * comprehension actually happens).
 *
 * The camera moves only between operations and is locked during them. Holds are load
 * bearing: the instinct to fill them with motion is exactly what makes explanatory
 * animation unreadable.
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
  b.at(0).add('ambient', 2.0, 0, 1, smooth);

  // ---------------------------------------------------------------- Stage 1
  openStage('input', 'Your digit', 'This is exactly what you handed the network.');

  b.at(0.15).add('s1.digit', 0.95, 0, 1, expoOut);
  b.at(0.15).add('s1.scale', 1.1, 0.84, 1, emphasizedOut);
  b.at(0.3).cue('enter');
  b.at(2.4);
  closeStage();

  // ---------------------------------------------------------------- Stage 2
  openStage(
    'prepare',
    'Preparing the input',
    'It gets cropped, shrunk to a 20-pixel box and re-centred on its centre of mass, exactly the way MNIST was built.',
  );

  b.at(2.5).add('s2.cropRing', 0.5, 0, 1, expoOut).cue('tick');
  b.at(2.5).add('s1.digit', 0.4, 1, 0.55, smooth);

  b.at(3.4).add('s2.boxed', 0.75, 0, 1, emphasized).cue('whoosh');
  b.at(3.4).add('s2.source', 0.6, 1, 0, smooth);
  b.at(3.4).add('s2.cropRing', 0.5, 1, 0, accelerateIn);

  b.at(4.5).add('s2.comDot', 0.4, 0, 1, expoOut).cue('tick');
  b.at(4.95).add('s2.centre', 0.7, 0, 1, emphasized);
  b.at(5.4).add('s2.comDot', 0.5, 1, 0, smooth);

  b.at(5.65).add('s2.input', 0.65, 0, 1, emphasizedOut);
  b.at(5.65).add('s2.boxed', 0.5, 1, 0, smooth);
  b.at(5.85).add('s2.grid', 0.7, 0, 1, smooth).cue('settle');
  b.at(6.8);
  closeStage();

  // ---------------------------------------------------------------- Stage 3
  openStage(
    'edges',
    'First look',
    'Eight small filters sweep across the image. Each one lights up where it finds the edge it was trained to spot.',
  );

  // Camera travels between operations, never during one.
  b.at(6.8).add('cam.station', 1.25, 0, 1, emphasized);
  // Straight to zero, not merely dim: the camera passes z=0 on the way, and anything
  // still alive there ends up between the lens and the subject.
  b.at(6.8).add('s2.input', 0.5, 1, 0, smooth);
  b.at(6.8).add('s2.grid', 0.5, 1, 0, smooth);

  // The kernels themselves: the actual trained weights, not an illustration.
  b.at(7.45).stagger((i) => `s3.kernel${i}`, C1, 0.45, 0, 1, 0.055, expoOut);
  b.at(8.0).cue('tick');

  // One hero filter reads across the plate at full fidelity. Showing all eight sweeping
  // at once is unreadable; the rest resolve afterwards as an already-finished grid.
  b.at(8.15).add('s3.hero', 0.5, 0, 1, expoOut);
  // Push in for the sweep. Framed for the whole eight-plate grid, one filter reading the
  // image is a small event in a big empty frame.
  b.at(8.15).add('cam.zoom', 0.95, 1, 0.62, emphasized);
  b.at(8.15).add('cam.centerAdjust', 0.95, 0, -0.6, emphasized);
  // The working filter comes down out of the row to sit beside its own output.
  b.at(8.15).add('s3.heroLift', 0.7, 0, 1, emphasized);
  b.at(10.8).add('s3.heroLift', 0.7, 1, 0, smooth);
  b.at(8.3).add('s3.sweep', 1.7, 0, 1, smooth).cue('sweep');
  b.at(8.3).add('s3.heroKernel', 1.7, 0, 1, smooth);
  // Negative responses appear as they are computed. Revealing them later would imply
  // ReLU had already run, which is the wrong order.
  b.at(8.35).add('s3.signed', 0.5, 0, 1, smooth);

  // The other seven, staggered by how strongly each fired rather than by index, so the
  // order of appearance is itself information.
  b.at(10.8).stagger((i) => `s3.plate${i}`, C1, 0.5, 0, 1, 0.05, expoOut, run.conv1Order);
  b.at(10.8).add('s3.spread', 0.9, 0, 1, emphasizedOut).cue('bloom');
  b.at(10.8).add('cam.zoom', 0.95, 0.62, 1, emphasized);
  b.at(10.8).add('cam.centerAdjust', 0.95, -0.6, 0, emphasized);

  // ReLU. Anticipation first: the negative half brightens so the eye is already on it
  // when it goes. Then the cut. The positive half is untouched throughout — that
  // asymmetry is the whole lesson, so nothing about it may change here.
  b.at(12.9).add('s3.reluHint', 0.45, 0, 1, smooth).cue('reveal');
  b.at(13.7).add('s3.relu', 0.7, 0, 1, emphasized).cue('cut');
  b.at(13.7).add('s3.reluHint', 0.55, 1, 0, smooth);
  b.at(14.9);
  closeStage();

  // ---------------------------------------------------------------- Stage 4
  openStage(
    'shapes',
    'Finding shapes',
    'Neighbouring cells compete and only the strongest survives, so the picture halves. Then the survivors get combined into richer features.',
  );

  // Come back to one map and look closely. At grid scale a 2x2 window is two pixels
  // wide, so pooling is literally invisible unless we zoom in for it.
  b.at(14.9).add('s4.focus', 0.9, 0, 1, emphasized);
  b.at(14.9).add('cam.zoom', 0.9, 1, 0.76, emphasized);
  b.at(14.9).add('cam.centerAdjust', 0.9, 0, -0.66, emphasized);
  b.at(18.3).add('cam.zoom', 0.8, 0.76, 1, smooth);
  b.at(18.3).add('cam.centerAdjust', 0.8, -0.66, 0, smooth);
  b.at(16.3).add('s4.contest', 0.3, 0, 1, smooth).cue('contest');
  b.at(16.6).add('s4.contest', 0.25, 1, 0, smooth);
  b.at(16.65).add('s4.contract', 0.9, 0, 1, emphasized).cue('contract');

  b.at(18.3).add('s4.focus', 0.6, 1, 0, smooth);
  b.at(18.3).add('cam.station', 1.35, 1, 2, emphasized);
  b.at(18.4).add('s3.fade', 0.8, 1, 0, smooth);

  b.at(19.2).stagger((i) => `s4.plate${i}`, C2, 0.45, 0, 1, 0.038, expoOut, run.conv2Order);
  b.at(19.2).cue('bloom');

  // Selective luminance: the maps that matter stay lit, the rest recede. The strongest
  // guidance device available, and it costs nothing.
  b.at(20.9).add('s4.rank', 0.75, 0, 1, smooth).cue('reveal');
  b.at(22.0).add('s4.pool', 0.7, 0, 1, emphasized).cue('contract');
  b.at(22.9);
  closeStage();

  // ---------------------------------------------------------------- Stage 5
  openStage(
    'matching',
    'Matching possibilities',
    'Every surviving feature votes through 32 hidden units. Cyan argues for a digit, coral argues against it.',
  );

  b.at(22.9).add('cam.station', 1.4, 2, 3, emphasized);
  b.at(22.9).add('s4.fade', 0.9, 1, 0, smooth);

  b.at(23.7).add('s5.block', 0.7, 0, 1, expoOut);
  b.at(24.0).stagger((i) => `s5.unit${i}`, HIDDEN, 0.4, 0, 1, 0.018, expoOut);
  b.at(24.7).add('s5.flowA', 0.7, 0, 1, smooth).cue('flow');

  b.at(25.4).stagger((i) => `s5.cand${i}`, CLASSES, 0.45, 0, 1, 0.05, expoOut, run.ranked);
  b.at(26.1).add('s5.flowB', 0.8, 0, 1, smooth).cue('flow');
  b.at(26.9).add('s5.weigh', 0.9, 0, 1, emphasizedOut).cue('reveal');
  b.at(29.0);
  closeStage();

  // ---------------------------------------------------------------- Stage 6
  openStage(
    'decision',
    'The decision',
    'Ten raw scores become a share of belief: stretch the gaps with an exponential, then split one single unit of certainty between them.',
  );

  b.at(29.0).add('s6.gather', 1.0, 0, 1, emphasized).cue('whoosh');
  b.at(29.0).add('s5.flowA', 0.5, 1, 0, smooth);
  // All the way to zero: leftover particles drifting across the bars in stage 6 read as
  // noise, and this beat needs an uncluttered frame to be legible.
  b.at(29.0).add('s5.flowB', 0.6, 1, 0, smooth);

  // The raw logits. Some are negative and they sum to nothing in particular.
  b.at(30.0).add('s6.logits', 0.75, 0, 1, emphasizedOut).cue('tick');

  // Exponentiate: small gaps become large ones. Dramatic for free, and honest.
  b.at(31.5).add('s6.exp', 0.85, 0, 1, emphasized).cue('stretch');

  // Normalise: one fixed unit of light divided ten ways.
  b.at(33.0).add('s6.budget', 0.4, 0, 1, smooth);
  b.at(33.0).add('s6.normalize', 0.9, 0, 1, emphasized).cue('pour');

  // The winner locks. Losers may overshoot; the winner may not — certainty should read
  // as a lock, not a wobble.
  b.at(33.95).add('s6.lock', 0.75, 0, 1, lock).cue('lock');
  b.at(33.95).add('post.aberration', 0.8, 0.006, 0, smooth);
  b.at(34.8);
  closeStage();

  // ---------------------------------------------------------------- Stage 7
  openStage('answer', 'The answer', '');

  b.at(34.8).add('cam.station', 1.25, 3, 4, emphasized);
  b.at(34.8).add('s6.fade', 0.9, 1, 0, smooth);
  b.at(35.0).add('post.defocus', 0.7, 0.25, 0, expoOut);

  b.at(35.5).add('s7.digit', 0.85, 0, 1, emphasizedOut).cue('arrive');
  b.at(36.1).add('s7.saliency', 0.95, 0, 1, smooth).cue('reveal');
  b.at(37.1).add('s7.counter', 0.95, 0, 1, smooth);
  b.at(38.1).add('s7.counter', 0.8, 1, 0.35, smooth);
  b.at(39.2);
  closeStage();

  return { timeline: b.build(39.2), stages };
}

/** Which stage contains time `t`. */
export function stageAt(stages: StageInfo[], t: number): StageInfo {
  for (let i = stages.length - 1; i >= 0; i--) {
    if (t >= stages[i].start) return stages[i];
  }
  return stages[0];
}
