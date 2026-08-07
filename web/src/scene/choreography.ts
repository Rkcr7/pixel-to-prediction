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
 * reading the annotation text, not from the animation. So: every stage answers a why, and
 * a claim stays legible for `words / 2.5` seconds and outlives the beat it describes.
 *
 * Total length is about 56 seconds, and that number is set by reading, not by animation.
 * Nine claims at 2.5 words per second need 32 seconds of screen time between them; below
 * roughly 52 seconds they begin to overlap each other, which is what "too fast" actually
 * feels like from the viewer's side. Getting under a minute meant deleting claims, not
 * shortening holds: anything the picture already states with numbers on it (the multiply,
 * the sum, the bias, the flatten) lost its caption and kept its drawing.
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
 * below silent-reading speed. Anything shorter is decoration that happens to contain
 * words. Every claim here is eleven words or fewer, which is what lets its hold fit
 * inside one beat instead of running over the next one.
 */
function note(b: TimelineBuilder, key: string, at: number, text: string) {
  const words = text.trim().split(/\s+/).length;
  const hold = Math.max(3.4, words / 2.5);
  b.at(at).add(key, 0.45, 0, 1, smooth);
  b.at(at + 0.45 + hold).add(key, 0.5, 1, 0, smooth);
  return hold;
}

/**
 * Every claim in the piece, in running order, so the copy can be edited in one place.
 *
 * What is absent matters as much as what is here. There is no caption for the multiply,
 * the running sum, the bias, or the flatten, because each of those is drawn with its own
 * live numbers beside it; a sentence restating a number the viewer is already looking at
 * costs four seconds and teaches nothing.
 */
export const NOTES: Record<string, string> = {
  'n.numbers': 'Not a picture. Just 784 numbers.',
  'n.share': 'The filter never changed as it moved. Same 25 numbers everywhere.',
  'n.relu': 'Negatives deleted. Positives left completely untouched.',
  'n.template': 'Each unit holds a picture of what it wants to see.',
  'n.exp': 'The exponential exaggerates the gaps before they are shared out.',
  // Nothing for stage 7. The answer card replaces the story panel the moment it lands, so
  // a claim written for that stage would play into a slot nobody can see. Its closing
  // thought lives in the card itself, as `.result__origin`.
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
  b.at(0).add('post.fade', 0.8, 0, 1, smooth);
  b.at(0).add('post.aberration', 1.4, 0.018, 0.006, smooth);
  b.at(0).add('post.groundGlow', 1.6, 0, 0.34, smooth);
  b.at(0).add('post.exposure', 1.2, 0.75, 1.0, smooth);
  b.at(0).add('post.defocus', 1.1, 0.35, 0, expoOut);

  // ---------------------------------------------------------------- Stage 1
  // No claim here. The caption says the whole of it, and a second line repeating that
  // the network sees something smaller only delays the first actual transformation.
  openStage('input', 'Your digit', 'This is exactly what you handed the network.');

  b.at(0.2).add('s1.digit', 1.1, 0, 1, expoOut);
  b.at(0.2).add('s1.scale', 1.3, 0.84, 1, emphasizedOut);
  b.at(0.4).cue('enter');
  b.at(3.0);
  closeStage();

  // ---------------------------------------------------------------- Stage 2
  // BUT: it cannot use that directly.
  openStage(
    'prepare',
    'Preparing the input',
    'It cannot use that directly. The drawing gets cropped, shrunk into a 20-pixel box, and re-centred on its centre of mass.',
  );

  b.at(3.2).add('s2.cropRing', 0.6, 0, 1, expoOut).cue('tick');
  b.at(3.2).add('s1.digit', 0.5, 1, 0.55, smooth);

  b.at(4.3).add('s2.boxed', 0.85, 0, 1, emphasized).cue('whoosh');
  b.at(4.3).add('s2.source', 0.7, 1, 0, smooth);
  b.at(4.3).add('s2.cropRing', 0.55, 1, 0, accelerateIn);

  b.at(5.55).add('s2.comDot', 0.45, 0, 1, expoOut).cue('tick');
  b.at(6.2).add('s2.centre', 0.85, 0, 1, emphasized);
  b.at(6.9).add('s2.comDot', 0.55, 1, 0, smooth);

  b.at(7.3).add('s2.input', 0.7, 0, 1, emphasizedOut);
  b.at(7.3).add('s2.boxed', 0.55, 1, 0, smooth);
  b.at(7.5).add('s2.grid', 0.8, 0, 1, smooth).cue('settle');
  // Deliberately outlives the stage. It is the claim the whole of stage 3 rests on, and
  // it is still true of what is on screen while the camera travels to the filters.
  note(b, 'n.numbers', 7.65, NOTES['n.numbers']);
  b.at(10.6);
  closeStage();

  // ---------------------------------------------------------------- Stage 3
  // THEREFORE: now it can look.
  openStage(
    'edges',
    'First look',
    'Now it can look. Eight small filters sweep across the image, and each one lights up wherever it finds the edge it was trained to spot.',
  );

  // Camera travels between operations, never during one.
  b.at(10.6).add('cam.station', 1.4, 0, 1, emphasized);
  // Straight to zero, not merely dim: the camera passes z=0 on the way, and anything
  // still alive there ends up between the lens and the subject.
  b.at(10.6).add('s2.input', 0.55, 1, 0, smooth);
  b.at(10.6).add('s2.grid', 0.55, 1, 0, smooth);

  // The kernels themselves: the actual trained weights, not an illustration.
  b.at(11.3).stagger((i) => `s3.kernel${i}`, C1, 0.5, 0, 1, 0.075, expoOut);
  b.at(11.9).cue('tick');

  // One hero filter reads across the plate at full fidelity. Showing all eight sweeping
  // at once is unreadable; the rest resolve afterwards as an already-finished grid.
  //
  // It arrives before the camera moves, not with it. The station is framed for a filter
  // row plus two rows of plates, so until something occupies the middle the shot is a
  // thin band of tiles across the top of an empty frame — and that empty frame is the
  // first thing anyone sees of the network actually working.
  b.at(12.2).add('s3.hero', 0.55, 0, 1, expoOut);
  // Push in for the sweep, and bring the aim down with it: the station centre is set
  // high to include the filter row, which is no longer the subject once we are close.
  b.at(12.9).add('cam.zoom', 1.1, 1, 0.78, emphasized);
  b.at(12.9).add('cam.centerAdjust', 1.1, 0, -0.55, emphasized);
  // The working filter comes down out of the row to sit beside its own output.
  b.at(12.9).add('s3.heroLift', 0.85, 0, 1, emphasized);

  b.at(13.2).add('s3.sweep', 2.5, 0, 1, smooth).cue('sweep');
  b.at(13.2).add('s3.heroKernel', 2.5, 0, 1, smooth);
  // Negative responses appear as they are computed. Revealing them later would imply
  // ReLU had already run, which is the wrong order.
  b.at(13.3).add('s3.signed', 0.55, 0, 1, smooth);

  // Timed to land while the filter is still visibly mid-traverse. The claim is about
  // the traverse; reading it after the sweep has finished makes it a caption on a
  // still picture instead of a description of the thing happening.
  note(b, 'n.share', 14.6, NOTES['n.share']);

  // The other seven, staggered by how strongly each fired rather than by index, so the
  // order of appearance is itself information.
  b.at(16.2).stagger((i) => `s3.plate${i}`, C1, 0.55, 0, 1, 0.085, expoOut, run.conv1Order);
  b.at(16.2).add('s3.spread', 1.1, 0, 1, emphasizedOut).cue('bloom');
  b.at(16.2).add('cam.zoom', 1.1, 0.78, 1, emphasized);
  b.at(16.2).add('cam.centerAdjust', 1.1, -0.55, 0, emphasized);
  b.at(16.2).add('s3.heroLift', 0.8, 1, 0, smooth);

  // ReLU. Anticipation first: the negative half brightens so the eye is already on it
  // when it goes. Then the cut. The positive half is untouched throughout — that
  // asymmetry is the whole lesson, so nothing about it may change here.
  b.at(18.3).add('s3.reluHint', 0.6, 0, 1, smooth).cue('reveal');
  b.at(19.3).add('s3.relu', 0.85, 0, 1, emphasized).cue('cut');
  b.at(19.3).add('s3.reluHint', 0.7, 1, 0, smooth);
  note(b, 'n.relu', 19.9, NOTES['n.relu']);
  b.at(22.0);
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
  b.at(22.0).add('s4.focus', 1.0, 0, 1, emphasized);
  b.at(22.0).add('cam.zoom', 1.0, 1, 0.78, emphasized);
  b.at(22.0).add('cam.centerAdjust', 1.0, 0, -0.62, emphasized);

  b.at(23.3).add('s4.contest', 0.35, 0, 1, smooth).cue('contest');
  b.at(23.65).add('s4.contest', 0.3, 1, 0, smooth);
  b.at(23.8).add('s4.contract', 1.0, 0, 1, emphasized).cue('contract');
  // No caption for the contest. The scene already carries "each 2 by 2 block keeps only
  // its brightest cell" anchored directly to the grid it is about, and repeating it in
  // the panel would print the same sentence twice on one screen while displacing the
  // stage's own explanation.

  // Leave this station in the arrangement it is already in.
  //
  // Releasing s4.focus would send the hero plate back to its grid slot AND restore the
  // other seven from 4% opacity to full, right as the camera starts moving. Eight plates
  // popping back to full brightness while the lens rushes past them is not a transition.
  b.at(26.3).add('s3.fade', 0.9, 1, 0, smooth);
  b.at(26.9).add('cam.station', 1.45, 1, 2, emphasized);
  b.at(26.9).add('cam.zoom', 0.95, 0.78, 1, smooth);
  b.at(26.9).add('cam.centerAdjust', 0.95, -0.62, 0, smooth);
  // Only now is it safe to reset, with the station already invisible.
  b.at(28.0).add('s4.focus', 0.3, 1, 0, smooth);

  // Only once the camera has actually arrived. Staggering these in mid-flight makes
  // them appear off-centre and then slide, which reads as a glitch rather than a reveal.
  b.at(28.6).stagger((i) => `s4.plate${i}`, C2, 0.5, 0, 1, 0.045, expoOut, run.conv2Order);
  b.at(28.6).cue('bloom');

  // Selective luminance: the maps that matter stay lit, the rest recede.
  b.at(30.0).add('s4.rank', 0.85, 0, 1, smooth).cue('reveal');
  b.at(30.7).add('s4.pool', 0.8, 0, 1, emphasized).cue('contract');
  b.at(31.5);
  closeStage();

  // ---------------------------------------------------------------- Stage 5
  // THEREFORE: those features are now just numbers.
  openStage(
    'matching',
    'Matching possibilities',
    'Those features are now just 784 numbers. Every one of them votes, through 32 hidden units, on all ten digits at once.',
  );

  b.at(31.5).add('cam.station', 1.5, 2, 3, emphasized);
  b.at(31.5).add('s4.fade', 0.95, 1, 0, smooth);

  b.at(32.5).add('s5.block', 0.8, 0, 1, expoOut);

  // The dense layer drawn as an actual dot product. Without this the 32 lit and unlit
  // sockets that follow are unexplained decoration.
  b.at(33.7).add('s5.panels', 1.15, 0, 1, emphasized).cue('whoosh');
  // Push in: three panels of detail framed for the whole dense station are too small to
  // read the cells that carry the entire point.
  b.at(33.7).add('cam.zoom', 1.15, 1, 0.82, emphasized);
  // The only claim in this stage. What the middle panel is, and why the left one exists,
  // is the part no picture can state on its own; everything after it (multiply, sum,
  // bias, threshold) arrives with its own live numbers attached.
  note(b, 'n.template', 34.3, NOTES['n.template']);

  b.at(37.0).add('s5.agree', 0.9, 0, 1, emphasizedOut).cue('reveal');
  b.at(38.5).add('s5.sum', 1.1, 0, 1, emphasized).cue('pour');
  b.at(39.5).add('s5.gate', 0.9, 0, 1, emphasized).cue('tick');

  // Now the lattice means something.
  b.at(40.9).add('s5.lattice', 1.0, 0, 1, smooth);
  b.at(40.9).add('cam.zoom', 1.0, 0.82, 1, smooth);
  b.at(40.9).stagger((i) => `s5.unit${i}`, HIDDEN, 0.45, 0, 1, 0.025, expoOut);

  b.at(42.1).add('s5.flowA', 0.8, 0, 1, smooth).cue('flow');
  b.at(42.6).stagger((i) => `s5.cand${i}`, CLASSES, 0.5, 0, 1, 0.06, expoOut, run.ranked);
  b.at(43.2).add('s5.flowB', 0.9, 0, 1, smooth).cue('flow');
  b.at(43.8).add('s5.weigh', 1.0, 0, 1, emphasizedOut).cue('reveal');
  b.at(44.9);
  closeStage();

  // ---------------------------------------------------------------- Stage 6
  // BUT: those ten scores do not add up to anything.
  openStage(
    'decision',
    'The decision',
    'But those ten scores do not add up to anything yet. Stretch the gaps with an exponential, then split one single unit of certainty between them.',
  );

  b.at(44.9).add('s6.gather', 1.15, 0, 1, emphasized).cue('whoosh');
  // This station shares the dense layer's frame, which is sized for a 32-unit lattice and
  // a column of ten candidates. The bars use about half of it, so without a push in and a
  // small lift of the aim, the whole decision plays out in the top two thirds of an
  // otherwise empty screen.
  b.at(44.9).add('cam.zoom', 1.2, 1, 0.88, emphasized);
  b.at(44.9).add('cam.centerAdjust', 1.2, 0, 0.45, emphasized);
  b.at(44.9).add('s5.flowA', 0.55, 1, 0, smooth);
  // All the way to zero: leftover particles drifting across the bars read as noise, and
  // this beat needs an uncluttered frame to be legible.
  b.at(44.9).add('s5.flowB', 0.65, 1, 0, smooth);

  // The raw logits. Some are negative and they sum to nothing in particular.
  b.at(46.1).add('s6.logits', 0.85, 0, 1, emphasizedOut).cue('tick');

  // Exponentiate: small gaps become large ones. Dramatic for free, and honest.
  b.at(47.6).add('s6.exp', 0.95, 0, 1, emphasized).cue('stretch');
  note(b, 'n.exp', 48.0, NOTES['n.exp']);

  // Normalise: one fixed unit of light divided ten ways. The container arrives on the
  // SAME clip, not before it: the landmark appearing exactly as the units change is what
  // makes the unit change legible.
  b.at(50.1).add('s6.budget', 1.0, 0, 1, emphasized);
  b.at(50.1).add('s6.normalize', 1.0, 0, 1, emphasized).cue('pour');

  // The winner locks. Losers may overshoot; the winner may not — certainty should read
  // as a lock, not a wobble.
  b.at(51.1).add('s6.lock', 0.8, 0, 1, lock).cue('lock');
  b.at(51.1).add('post.aberration', 0.9, 0.006, 0, smooth);
  b.at(51.9);
  closeStage();

  // ---------------------------------------------------------------- Stage 7
  openStage('answer', 'The answer', '');

  b.at(51.9).add('cam.station', 1.35, 3, 4, emphasized);
  b.at(51.9).add('cam.zoom', 1.35, 0.88, 1, emphasized);
  b.at(51.9).add('cam.centerAdjust', 1.35, 0.45, 0, emphasized);
  b.at(51.9).add('s6.fade', 1.0, 1, 0, smooth);
  b.at(52.1).add('post.defocus', 0.8, 0.25, 0, expoOut);

  b.at(52.6).add('s7.digit', 0.95, 0, 1, emphasizedOut).cue('arrive');
  b.at(53.4).add('s7.saliency', 1.05, 0, 1, smooth).cue('reveal');
  b.at(54.4).add('s7.counter', 1.05, 0, 1, smooth);
  b.at(55.2).add('s7.counter', 0.9, 1, 0.35, smooth);
  // A couple of seconds of stillness on the answer. Nothing is moving, which is the
  // point: this is the frame people screenshot.
  b.at(57.2);
  closeStage();

  return { timeline: b.build(57.2), stages };
}

/** Which stage contains time `t`. */
export function stageAt(stages: StageInfo[], t: number): StageInfo {
  for (let i = stages.length - 1; i >= 0; i--) {
    if (t >= stages[i].start) return stages[i];
  }
  return stages[0];
}
