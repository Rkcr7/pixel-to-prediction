/**
 * The colour language.
 *
 * Two encodings, kept strictly orthogonal because conflating them is the classic mistake:
 *
 *   MAGNITUDE — a magma-like ramp with monotonically rising lightness. Its zero end is
 *   pure black so that additive blending needs no alpha channel: an inactive cell simply
 *   contributes nothing. Pure white is reserved for the maximum, so the eye always knows
 *   where the peak is.
 *
 *   SIGN — cyan for excitatory, coral for inhibitory, at matched lightness so that sign
 *   never masquerades as strength. Used only where weights have a sign.
 *
 * Deliberately not the red/white/blue diverging scheme every paper figure uses, and
 * deliberately not jet. Saturated thin red is also avoided on purpose: 4:2:0 chroma
 * subsampling mangles it, and this is designed to survive being screen-recorded and
 * reposted.
 */

export interface Anchor {
  t: number;
  hex: string;
}

/** Magnitude ramp. Black-anchored so zero activation is invisible under additive blending. */
export const MAGNITUDE_ANCHORS: Anchor[] = [
  { t: 0.0, hex: '#000000' },
  { t: 0.08, hex: '#0B0620' },
  { t: 0.2, hex: '#241056' },
  { t: 0.34, hex: '#4A1478' },
  { t: 0.47, hex: '#7A1B7E' },
  { t: 0.6, hex: '#B02A72' },
  { t: 0.72, hex: '#DC4A56' },
  { t: 0.83, hex: '#EE7B32' },
  { t: 0.92, hex: '#F8B33C' },
  { t: 0.97, hex: '#FDE28C' },
  { t: 1.0, hex: '#FFFFFF' },
];

export const COLORS = {
  /** Page and scene ground. Not pure black: a cool near-black gives bloom something to
   *  sit against and keeps dark gradients from banding. */
  ground: '#05060A',
  groundLift: '#0A0C14',
  /** Excitatory. */
  positive: '#22D3EE',
  /** Inhibitory. Coral rather than red, for compression and for the palette's warmth. */
  negative: '#FB7185',
  /** Chrome must never be as bright as the dimmest live signal. */
  chrome: '#8A93A6',
  chromeDim: '#4A5266',
  text: '#E8ECF5',
  accent: '#F8B33C',
} as const;

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

/**
 * Build a 256-entry sRGB lookup table. Interpolating in sRGB (rather than linear light)
 * is intentional: it keeps the perceptual steps even, which is what a reader is judging.
 * The shader converts to linear before any HDR maths happens.
 */
export function buildRamp(anchors: Anchor[], size = 256): Uint8Array {
  const out = new Uint8Array(size * 4);
  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    let a = anchors[0];
    let b = anchors[anchors.length - 1];
    for (let k = 0; k < anchors.length - 1; k++) {
      if (t >= anchors[k].t && t <= anchors[k + 1].t) {
        a = anchors[k];
        b = anchors[k + 1];
        break;
      }
    }
    const span = b.t - a.t;
    const f = span <= 0 ? 0 : (t - a.t) / span;
    const ca = hexToRgb(a.hex);
    const cb = hexToRgb(b.hex);
    out[i * 4 + 0] = Math.round((ca[0] + (cb[0] - ca[0]) * f) * 255);
    out[i * 4 + 1] = Math.round((ca[1] + (cb[1] - ca[1]) * f) * 255);
    out[i * 4 + 2] = Math.round((ca[2] + (cb[2] - ca[2]) * f) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** CSS colour for a normalised magnitude, for DOM elements that must match the canvas. */
export function magnitudeCss(t: number, ramp: Uint8Array): string {
  const i = Math.min(255, Math.max(0, Math.round(t * 255)));
  return `rgb(${ramp[i * 4]}, ${ramp[i * 4 + 1]}, ${ramp[i * 4 + 2]})`;
}
