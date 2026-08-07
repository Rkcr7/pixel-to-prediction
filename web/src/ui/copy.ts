/**
 * Turning measured evidence into a sentence.
 *
 * The rule: never claim more than the numbers support. When attribution is diffuse or
 * confidence is low, say that instead of inventing a tidy reason. A canned explanation
 * that is right by luck is worse than an honest "this one was close".
 *
 * Phrasing is chosen deterministically from the evidence, so the same drawing always
 * produces the same words. Randomised copy would read as a slot machine.
 */

import type { Evidence } from '../core/engine';

/** Mirrors REGION_NAMES in the Rust explain module, row-major from the top-left. */
const REGIONS = [
  'up in the top left',
  'across the top',
  'up in the top right',
  'down the left side',
  'through the middle',
  'down the right side',
  'in the bottom left',
  'along the bottom',
  'in the bottom right',
];

const REGION_SHORT = [
  'top left',
  'top',
  'top right',
  'left',
  'centre',
  'right',
  'bottom left',
  'bottom',
  'bottom right',
];

export interface Explanation {
  /** The main sentence: why this answer. */
  reason: string;
  /** The runner-up sentence, or empty when there is nothing interesting to say. */
  counter: string;
  /** A short confidence phrase for the headline. */
  confidence: string;
  /** True when the model is genuinely unsure and the UI should soften its tone. */
  hedged: boolean;
}

function holePhrase(holes: number): string {
  if (holes >= 2) return 'two closed loops stacked on top of each other';
  if (holes === 1) return 'a closed loop';
  return '';
}

function confidencePhrase(p: number): string {
  if (p >= 0.999) return 'no doubt at all';
  if (p >= 0.99) return 'very sure';
  if (p >= 0.92) return 'confident';
  if (p >= 0.75) return 'fairly sure';
  if (p >= 0.55) return 'leaning that way';
  return 'genuinely unsure';
}

export function explain(e: Evidence): Explanation {
  const hedged = e.p1 < 0.7;
  const confidence = confidencePhrase(e.p1);

  const loop = holePhrase(e.holes);
  const where = REGIONS[e.strongestRegion] ?? 'through the middle';

  let reason: string;
  if (e.diffuse) {
    // No region carries a clear majority of the attribution. Say so.
    reason = loop
      ? `The evidence is spread right across the stroke, though ${loop} is doing some of the work.`
      : 'No single part of the stroke decided this one. The evidence is spread thin across the whole shape.';
  } else if (loop) {
    reason = `It leaned on ${loop} and the weight of ink ${where}.`;
  } else {
    reason = `It leaned on the ink ${where}, with no closed loop anywhere in the stroke.`;
  }

  // The counterfactual: which ink argued for the runner-up. Only worth saying when the
  // decision was actually close, otherwise it invents drama that was not there.
  let counter = '';
  if (e.top2 !== e.top1) {
    const contested = REGION_SHORT[e.contestedRegion] ?? 'centre';
    if (e.p2 > 0.02) {
      counter = `Its second guess was ${e.top2}. The ink around the ${contested} is what argued for that, and it lost.`;
    } else if (e.margin < 6) {
      counter = `Nothing else came close. ${e.top2} was next, and it was not near.`;
    }
  }

  return { reason, counter, confidence, hedged };
}

/** A short line for the moment the answer lands. */
export function verdict(e: Evidence): string {
  if (e.p1 >= 0.999) return `That is a ${e.top1}.`;
  if (e.p1 >= 0.9) return `It says ${e.top1}.`;
  if (e.p1 >= 0.6) return `It thinks ${e.top1}.`;
  return `Its best guess is ${e.top1}.`;
}

export function formatConfidence(p: number): string {
  const pct = p * 100;
  if (pct >= 99.95) return '>99.9%';
  if (pct >= 99) return `${pct.toFixed(1)}%`;
  return `${pct.toFixed(0)}%`;
}
