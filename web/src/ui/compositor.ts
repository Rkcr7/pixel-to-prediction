/**
 * The frame that gets recorded.
 *
 * `canvas.captureStream()` captures one canvas and nothing else, and every word in this
 * app is DOM layered over the WebGL canvas: the annotations, the story panel, the answer.
 * So a recorded clip had the scene moving with no explanation attached to it, which is
 * most of what the piece is for.
 *
 * This paints the scene and the text into a second canvas, and the recorder captures that
 * instead. It is not a screenshot of the page: the transport, the buttons and the browser
 * chrome are deliberately absent, because a clip with a "Record clip" button frozen inside
 * it is not something anyone would post. What it adds is the mark, so a clip that travels
 * on its own still says where it came from.
 */

import type { Label } from './annotations';

/** Where a text block sits, in the app's CSS pixels. */
export interface Rect {
  left: number;
  top: number;
  width: number;
}

export interface CaptionBlock {
  step: string;
  title: string;
  caption: string;
  /**
   * The story panel's own measured box.
   *
   * Taken from the DOM rather than derived, because the app has already solved where this
   * goes against the camera's reserved gutter, in both orientations, at every viewport
   * size. Recomputing it here produced a caption that ran over the first panel: two
   * layouts for one thing, and only one of them audited.
   */
  rect: Rect;
}

export interface AnswerBlock {
  digit: string;
  confidence: string;
  phrase: string;
  reason: string;
  counter: string;
  fool: string;
  rect: Rect;
}

const FONT =
  "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, system-ui, sans-serif";

const TEXT = '#F2F4F8';
const DIM = '#9AA6BC';
const FAINT = '#6C7891';
const ACCENT = '#F8B33C';
const CYAN = '#3EC9DA';
const GROUND = '#05060A';

export class FrameCompositor {
  readonly canvas = document.createElement('canvas');
  private readonly ctx: CanvasRenderingContext2D;
  /** Scale factor from the app's CSS pixels to the capture buffer. */
  private k = 1;

  constructor() {
    const ctx = this.canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('2D canvas is not available for recording');
    this.ctx = ctx;
  }

  /** Match the capture buffer to the scene's, so the GL frame is copied 1:1. */
  sync(source: HTMLCanvasElement, cssWidth: number) {
    if (this.canvas.width !== source.width || this.canvas.height !== source.height) {
      this.canvas.width = source.width;
      this.canvas.height = source.height;
    }
    this.k = source.width / Math.max(1, cssWidth);
  }

  paint(
    source: HTMLCanvasElement,
    labels: readonly Label[],
    caption: CaptionBlock | null,
    answer: AnswerBlock | null,
    portrait: boolean,
  ) {
    const x = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;

    x.setTransform(1, 0, 0, 1, 0, 0);
    x.globalAlpha = 1;
    x.fillStyle = GROUND;
    x.fillRect(0, 0, W, H);
    // Same task as the GL draw, so the drawing buffer is still valid: the context is
    // created without preserveDrawingBuffer and is cleared once the browser composites.
    x.drawImage(source, 0, 0, W, H);

    this.paintLabels(labels, W, H);
    if (caption) this.paintCaption(caption, portrait);
    if (answer) this.paintAnswer(answer, portrait);
    this.paintMark();
  }

  // -- pieces ----------------------------------------------------------------

  private paintLabels(labels: readonly Label[], W: number, H: number) {
    const x = this.ctx;
    for (const label of labels) {
      if (label.opacity <= 0.02) continue;
      const tag = (label.kind ?? 'tag') !== 'value';
      const size = Math.round((tag ? 12.5 : 14) * this.k);

      x.save();
      x.globalAlpha = Math.min(1, label.opacity);
      x.font = `400 ${size}px ${FONT}`;
      x.fillStyle = tag ? DIM : TEXT;
      x.textBaseline = 'middle';
      // The same shadow the stylesheet gives these, which is what keeps a label legible
      // where it crosses a bright feature map.
      x.shadowColor = 'rgba(5, 6, 10, 0.95)';
      x.shadowBlur = 12 * this.k;

      const text = tag ? label.text.toUpperCase() : label.text;
      const cx = label.x * W + (label.dx ?? 0) * this.k;
      const cy = label.y * H + (label.dy ?? 0) * this.k;

      if (tag) {
        // 0.13em tracking, drawn per character. Canvas letterSpacing is not available
        // everywhere and this has to look identical wherever the clip is recorded.
        this.trackedText(text, cx, cy, size * 0.13);
      } else {
        x.textAlign = 'center';
        x.fillText(text, cx, cy);
      }
      x.restore();
    }
  }

  /** Centred text with per-character tracking. */
  private trackedText(text: string, cx: number, cy: number, tracking: number) {
    const x = this.ctx;
    const chars = [...text];
    const width = chars.reduce((w, ch) => w + x.measureText(ch).width + tracking, 0) - tracking;
    x.textAlign = 'left';
    let px = cx - width / 2;
    for (const ch of chars) {
      x.fillText(ch, px, cy);
      px += x.measureText(ch).width + tracking;
    }
  }

  private paintCaption(block: CaptionBlock, portrait: boolean) {
    const x = this.ctx;
    const k = this.k;
    const left = block.rect.left * k;
    const boxW = block.rect.width * k;
    const titleSize = Math.round((portrait ? 22 : 27) * k);
    const bodySize = Math.round((portrait ? 14 : 15.5) * k);
    const lh = Math.round(bodySize * 1.62);

    x.save();
    x.textAlign = 'left';
    x.textBaseline = 'alphabetic';

    x.font = `400 ${bodySize}px ${FONT}`;
    const lines = wrap(x, block.caption, boxW);
    let y = block.rect.top * k + Math.round(11 * k);

    x.fillStyle = FAINT;
    x.font = `400 ${Math.round(11 * k)}px ${FONT}`;
    this.tracked(block.step, left, y, Math.round(11 * k) * 0.16);
    y += Math.round(15 * k) + titleSize;

    x.fillStyle = TEXT;
    x.font = `560 ${titleSize}px ${FONT}`;
    x.fillText(block.title, left, y);
    y += Math.round(18 * k);

    x.fillStyle = DIM;
    x.font = `400 ${bodySize}px ${FONT}`;
    for (const line of lines) {
      y += lh;
      x.fillText(line, left, y);
    }
    x.restore();
  }

  /** Left-aligned tracked text, for the step counter. */
  private tracked(text: string, left: number, y: number, tracking: number) {
    const x = this.ctx;
    let px = left;
    for (const ch of [...text]) {
      x.fillText(ch, px, y);
      px += x.measureText(ch).width + tracking;
    }
  }

  private paintAnswer(a: AnswerBlock, portrait: boolean) {
    const x = this.ctx;
    const k = this.k;
    const left = a.rect.left * k;
    const boxW = a.rect.width * k;

    const digitSize = Math.round((portrait ? 58 : 72) * k);
    const confSize = Math.round(21 * k);
    const bodySize = Math.round(14.5 * k);
    const lh = Math.round(bodySize * 1.55);

    x.save();
    x.textAlign = 'left';
    x.textBaseline = 'alphabetic';

    x.font = `400 ${bodySize}px ${FONT}`;
    const reason = wrap(x, a.reason, boxW);
    const counter = a.counter ? wrap(x, a.counter, boxW) : [];
    const fool = a.fool ? wrap(x, a.fool, boxW) : [];
    let y = a.rect.top * k + digitSize;

    x.fillStyle = ACCENT;
    x.font = `600 ${digitSize}px ${FONT}`;
    x.shadowColor = 'rgba(248, 179, 60, 0.4)';
    x.shadowBlur = 40 * k;
    x.fillText(a.digit, left, y);
    const digitW = x.measureText(a.digit).width;
    x.shadowBlur = 0;

    x.fillStyle = TEXT;
    x.font = `560 ${confSize}px ${FONT}`;
    x.fillText(a.confidence, left + digitW + Math.round(16 * k), y - Math.round(26 * k));
    x.fillStyle = FAINT;
    x.font = `400 ${Math.round(12.5 * k)}px ${FONT}`;
    x.fillText(a.phrase, left + digitW + Math.round(16 * k), y - Math.round(6 * k));

    y += Math.round(26 * k);
    x.fillStyle = TEXT;
    x.font = `400 ${bodySize}px ${FONT}`;
    for (const line of reason) {
      y += lh;
      x.fillText(line, left, y);
    }
    x.fillStyle = DIM;
    for (const line of counter) {
      y += lh;
      x.fillText(line, left, y);
    }
    x.fillStyle = CYAN;
    for (const line of fool) {
      y += lh;
      x.fillText(line, left, y);
    }
    x.restore();
  }

  /**
   * The mark and the name, small, in the corner.
   *
   * A clip travels without the page around it, so without this there is nothing on it
   * saying what it is or where to find it.
   */
  private paintMark() {
    const x = this.ctx;
    const k = this.k;
    const s = Math.round(15 * k);
    const left = Math.round(26 * k);
    const top = Math.round(24 * k);

    x.save();
    x.globalAlpha = 0.85;
    // Four pixels and one answer, drawn directly rather than loaded, so a recording never
    // waits on a network fetch.
    const unit = s / 64;
    x.fillStyle = CYAN;
    for (const [rx, ry] of [
      [1, 49],
      [17, 49],
      [17, 33],
      [49, 49],
    ]) {
      roundRect(x, left + rx * unit, top + ry * unit, 14 * unit, 14 * unit, 3.5 * unit);
      x.fill();
    }
    x.fillStyle = ACCENT;
    roundRect(x, left + 33 * unit, top + 1 * unit, 14 * unit, 62 * unit, 4 * unit);
    x.fill();

    x.fillStyle = DIM;
    x.font = `500 ${Math.round(13 * k)}px ${FONT}`;
    x.textAlign = 'left';
    x.textBaseline = 'middle';
    x.fillText('Pixel to Prediction', left + s + Math.round(10 * k), top + s / 2);
    x.restore();
  }
}

function roundRect(
  x: CanvasRenderingContext2D,
  px: number,
  py: number,
  w: number,
  h: number,
  r: number,
) {
  x.beginPath();
  x.moveTo(px + r, py);
  x.arcTo(px + w, py, px + w, py + h, r);
  x.arcTo(px + w, py + h, px, py + h, r);
  x.arcTo(px, py + h, px, py, r);
  x.arcTo(px, py, px + w, py, r);
  x.closePath();
}

/** Greedy wrap against the measured width of the current font. */
function wrap(x: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  if (!text) return [];
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (x.measureText(next).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}
