/**
 * The drawing surface.
 *
 * Stroke width matters more than it looks like it should. MNIST digits carry a stroke
 * roughly a tenth of their bounding-box height once normalised, so a brush much thinner
 * or fatter than that lands the drawing outside the distribution the model was trained on
 * and accuracy falls off a cliff for reasons the user cannot see. The default here is
 * tuned to land in that band for a digit drawn at a natural size.
 */

import { CANVAS_RES, IMG } from '../scene/constants';

/**
 * Everything needed to map the network's 28x28 field back onto the pad.
 *
 * Preprocessing runs crop to the ink, scale into a 20-pixel box, then translate by an
 * integer offset to centre the mass. All three steps are invertible from what the engine
 * already reports, which is what lets a hint computed in the network's coordinates be
 * drawn in the user's.
 */
export interface HintMap {
  /** Ink bounding box in extraction pixels: x, y, w, h. */
  bbox: readonly [number, number, number, number];
  boxedW: number;
  boxedH: number;
  /** Integer translation applied when placing the box into the 28x28 field. */
  offset: readonly [number, number];
}

export interface Point {
  x: number;
  y: number;
}

export type Stroke = Point[];

/**
 * Otsu's method: the threshold that maximises between-class variance, i.e. the split
 * that best separates a two-population brightness histogram. This is what lets a photo
 * of ink on paper be separated without knowing anything about the lighting.
 */
function otsuThreshold(hist: Uint32Array, total: number): number {
  let sum = 0;
  for (let v = 0; v < 256; v++) sum += v * hist[v];

  let sumBackground = 0;
  let weightBackground = 0;
  let best = 0;
  let bestVariance = -1;

  for (let t = 0; t < 256; t++) {
    weightBackground += hist[t];
    if (weightBackground === 0) continue;
    const weightForeground = total - weightBackground;
    if (weightForeground === 0) break;

    sumBackground += t * hist[t];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const delta = meanBackground - meanForeground;
    const variance = weightBackground * weightForeground * delta * delta;

    if (variance > bestVariance) {
      bestVariance = variance;
      best = t;
    }
  }
  return best;
}

export class DrawSurface {
  private ctx: CanvasRenderingContext2D;
  private extractCanvas: HTMLCanvasElement;
  private extractCtx: CanvasRenderingContext2D;
  private strokes: Stroke[] = [];
  private current: Stroke | null = null;
  private pointerId: number | null = null;
  /**
   * An uploaded image lives on its own layer rather than being baked into the canvas.
   * Repainting clears and redraws from scratch every frame of a stroke, so anything not
   * held here would be wiped the moment the user drew on top of it.
   */
  private background: HTMLCanvasElement | null = null;
  /**
   * The flip hint, pre-rendered into pad coordinates.
   *
   * Its own layer for the same reason the upload is: repaint clears and redraws from
   * scratch on every pointer sample, so anything not held here would be erased the
   * instant the user started drawing on it, which is precisely when they need it.
   */
  private hint: HTMLCanvasElement | null = null;

  /** Backing resolution. Twice the extraction size so the visible stroke is smooth. */
  readonly res = CANVAS_RES * 2;
  brushWidth = 30;

  onChange: (() => void) | null = null;
  onStrokeEnd: (() => void) | null = null;

  constructor(readonly canvas: HTMLCanvasElement) {
    canvas.width = this.res;
    canvas.height = this.res;
    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    if (!ctx) throw new Error('2D canvas is not available');
    this.ctx = ctx;

    this.extractCanvas = document.createElement('canvas');
    this.extractCanvas.width = CANVAS_RES;
    this.extractCanvas.height = CANVAS_RES;
    const ex = this.extractCanvas.getContext('2d', { willReadFrequently: true });
    if (!ex) throw new Error('2D canvas is not available');
    this.extractCtx = ex;

    this.attach();
    this.repaint();
  }

  private attach() {
    const c = this.canvas;
    c.style.touchAction = 'none';

    c.addEventListener('pointerdown', (e) => {
      if (this.pointerId !== null) return;
      this.pointerId = e.pointerId;
      this.rect = c.getBoundingClientRect();
      // Capture can fail for a pointer the browser no longer tracks. Losing capture
      // only costs us events outside the canvas; it must never abort the stroke.
      try {
        c.setPointerCapture(e.pointerId);
      } catch {
        /* keep drawing without capture */
      }
      this.current = [this.toLocal(e)];
      this.strokes.push(this.current);
      this.repaint();
      this.onChange?.();
    });

    c.addEventListener('pointermove', (e) => {
      if (this.pointerId !== e.pointerId || !this.current) return;
      // One layout read per event, not one per coalesced sample. A 240Hz stylus can
      // deliver a dozen samples per event, and getBoundingClientRect forces a
      // synchronous layout every time.
      this.rect = c.getBoundingClientRect();
      // Coalesced events give a much smoother line on high-rate pointers. Some browsers
      // hand back an empty list, and taking that literally silently drops the entire
      // stroke, so fall back to the event itself whenever the list comes back empty.
      const coalesced =
        typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [];
      const events = coalesced.length > 0 ? coalesced : [e];
      for (const ev of events) this.current.push(this.toLocal(ev));
      this.repaint();
      this.onChange?.();
    });

    const end = (e: PointerEvent) => {
      if (this.pointerId !== e.pointerId) return;
      this.pointerId = null;
      this.current = null;
      this.onStrokeEnd?.();
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('lostpointercapture', end);
  }

  /** Cached canvas rect, refreshed once per pointer event rather than per sample. */
  private rect: DOMRect | null = null;

  private toLocal(e: PointerEvent | { clientX: number; clientY: number }): Point {
    const r = (this.rect ??= this.canvas.getBoundingClientRect());
    return {
      x: ((e.clientX - r.left) / r.width) * this.res,
      y: ((e.clientY - r.top) / r.height) * this.res,
    };
  }

  /**
   * Show where a stroke would change the network's mind.
   *
   * `field` is the 28x28 flip gradient, so it lives in the network's coordinates: the
   * ink cropped, scaled into a 20-pixel box, and translated to centre its mass. All three
   * steps are inverted here so the hint lands on the pad where the user would actually
   * draw, which is the only place it is any use.
   *
   * Positives only. A negative says "less ink here", and the pad has a brush and an undo
   * but no eraser, so showing both halves would offer an instruction half of which
   * cannot be followed. The copy says "add ink" for the same reason.
   */
  setHint(field: Float32Array | null, map: HintMap | null) {
    if (!field || !map || map.boxedW === 0 || map.boxedH === 0) {
      this.hint = null;
      this.repaint();
      return;
    }

    const layer = document.createElement('canvas');
    layer.width = this.res;
    layer.height = this.res;
    const g = layer.getContext('2d');
    if (!g) return;

    const [bx, by, bw, bh] = map.bbox;
    const [ox, oy] = map.offset;
    // Extraction pixels to pad pixels, then field cells to extraction pixels.
    const k = this.res / CANVAS_RES;
    const cw = (bw / map.boxedW) * k;
    const ch = (bh / map.boxedH) * k;

    g.filter = `blur(${Math.max(2, cw * 0.3)}px)`;
    for (let fy = 0; fy < IMG; fy++) {
      for (let fx = 0; fx < IMG; fx++) {
        const v = field[fy * IMG + fx];
        // A floor, because the gradient is non-zero almost everywhere and painting all
        // 784 cells would wash the pad out instead of pointing at anything.
        if (v <= 0.22) continue;
        g.fillStyle = `rgba(79, 216, 232, ${Math.min(0.55, (v - 0.22) * 0.85)})`;
        g.fillRect((bx + (fx - ox) * (bw / map.boxedW)) * k, (by + (fy - oy) * (bh / map.boxedH)) * k, cw, ch);
      }
    }

    this.hint = layer;
    this.repaint();
  }

  private repaint() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.res, this.res);
    // Under the ink: it is a suggestion about the drawing, not part of it.
    if (this.hint) ctx.drawImage(this.hint, 0, 0);
    if (this.background) ctx.drawImage(this.background, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.fillStyle = '#ffffff';
    ctx.lineWidth = this.brushWidth;

    for (const stroke of this.strokes) {
      if (stroke.length === 0) continue;
      if (stroke.length === 1) {
        ctx.beginPath();
        ctx.arc(stroke[0].x, stroke[0].y, this.brushWidth / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      // Quadratic through midpoints: a polyline of raw pointer samples shows visible
      // corners at every sample, especially on a slow pointer.
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      for (let i = 1; i < stroke.length - 1; i++) {
        const mx = (stroke[i].x + stroke[i + 1].x) / 2;
        const my = (stroke[i].y + stroke[i + 1].y) / 2;
        ctx.quadraticCurveTo(stroke[i].x, stroke[i].y, mx, my);
      }
      const last = stroke[stroke.length - 1];
      ctx.lineTo(last.x, last.y);
      ctx.stroke();
    }
  }

  get isEmpty(): boolean {
    return this.strokes.length === 0 && this.background === null;
  }

  clear() {
    this.strokes = [];
    this.current = null;
    this.background = null;
    // A hint belongs to one drawing. Carrying it onto a blank pad would point at ink
    // that is no longer there.
    this.hint = null;
    this.repaint();
    this.onChange?.();
  }

  undo() {
    // Once every stroke is gone, undo removes the uploaded image too, which is what
    // "undo" means to someone who just uploaded the wrong file.
    if (this.strokes.length > 0) this.strokes.pop();
    else this.background = null;
    this.repaint();
    this.onChange?.();
  }

  /** Replace the drawing with an image, auto-inverting a dark-on-light photo. */
  async loadImage(file: File | Blob): Promise<void> {
    const bitmap = await createImageBitmap(file);

    const layer = document.createElement('canvas');
    layer.width = this.res;
    layer.height = this.res;
    const ctx = layer.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D canvas is not available');

    this.strokes = [];

    // Fit inside the square, preserving aspect. Photos are rarely square and stretching
    // one distorts the stroke widths the model is sensitive to.
    const scale = Math.min(this.res / bitmap.width, this.res / bitmap.height);
    const w = bitmap.width * scale;
    const h = bitmap.height * scale;
    ctx.drawImage(bitmap, (this.res - w) / 2, (this.res - h) / 2, w, h);
    bitmap.close?.();

    const img = ctx.getImageData(0, 0, this.res, this.res);
    const d = img.data;

    // Luminance, and a median test to decide whether the ink is dark or light. Median
    // rather than mean because a photo of a digit is mostly background, and the median
    // is exactly the background level.
    const hist = new Uint32Array(256);
    let covered = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] < 8) continue;
      const lum = Math.round(0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]);
      hist[Math.min(255, Math.max(0, lum))]++;
      covered++;
    }
    let acc = 0;
    let median = 128;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (acc >= covered / 2) {
        median = v;
        break;
      }
    }
    const invert = median > 127;

    // Ink strength, then Otsu.
    //
    // A photo on paper does not separate at a fixed cut: after inverting, paper sits
    // around 55 and ink around 190, so a naive threshold treats the whole sheet as ink
    // and the digit's bounding box becomes the entire page. Otsu finds the split that
    // best separates the two populations in this particular image, whatever the
    // lighting, and soft-thresholding around it preserves the antialiased stroke edge.
    const ink = new Float32Array(this.res * this.res);
    const inkHist = new Uint32Array(256);
    let histCount = 0;
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const a = d[i + 3];
      const lum = a < 8 ? (invert ? 255 : 0) : 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
      const value = invert ? 255 - lum : lum;
      ink[p] = value;
      // Only the pixels the photo actually covers may vote.
      //
      // A non-square image is letterboxed into this square canvas, and the padding is
      // transparent. Feeding that padding to Otsu adds a third population at zero, and
      // the best two-way split then falls between the padding and the paper instead of
      // between the paper and the ink — which turns the whole sheet into ink.
      if (a >= 8) {
        inkHist[Math.min(255, Math.max(0, Math.round(value)))]++;
        histCount++;
      }
    }

    const cut = otsuThreshold(inkHist, histCount);
    // Soft band around the cut so a stroke keeps its grey edge rather than becoming a
    // hard-edged stencil, which would sit off the distribution the model was trained on.
    const lo = cut * 0.72;
    const hi = Math.min(255, cut * 1.18 + 12);

    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const t = (ink[p] - lo) / Math.max(hi - lo, 1);
      const clamped = Math.min(1, Math.max(0, t));
      d[i] = d[i + 1] = d[i + 2] = 255;
      d[i + 3] = Math.round(clamped * clamped * (3 - 2 * clamped) * 255);
    }
    ctx.putImageData(img, 0, 0);
    this.background = layer;
    this.repaint();
    this.onChange?.();
    this.onStrokeEnd?.();
  }

  /**
   * Extract ink as a CANVAS_RES square of floats in [0,1]. Downsampling through
   * `drawImage` uses the browser's own filtered resampler, which antialiases the stroke
   * the way MNIST's own normalisation did.
   */
  extract(): Float32Array {
    const ex = this.extractCtx;
    ex.clearRect(0, 0, CANVAS_RES, CANVAS_RES);
    ex.drawImage(this.canvas, 0, 0, CANVAS_RES, CANVAS_RES);
    const data = ex.getImageData(0, 0, CANVAS_RES, CANVAS_RES).data;
    const out = new Float32Array(CANVAS_RES * CANVAS_RES);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      // The surface is drawn as opaque white ink, so alpha is the ink coverage.
      out[p] = data[i + 3] / 255;
    }
    return out;
  }
}
