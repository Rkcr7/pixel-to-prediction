/**
 * Labels anchored to points in the 3D scene.
 *
 * They are DOM elements positioned from a projected world point rather than text drawn
 * into the GL target. Two reasons: the render target is often downscaled for fill rate
 * and type would go soft with it, and screen readers get nothing from a texture.
 *
 * Elements are pooled by id so a label keeps its node across frames and CSS transitions
 * behave instead of restarting.
 */

export interface Label {
  id: string;
  text: string;
  /** Viewport fractions, 0..1 from the top left. */
  x: number;
  y: number;
  opacity: number;
  /**
   * Three registers, and the distinction is load bearing.
   *
   * 'tag' names an object: uppercase, tracked, at most five words, anchored to a point
   * in the scene. 'value' is a tabular number. 'note' makes a claim: sentence case, one
   * sentence, and pinned to a fixed lower-third slot rather than tracking the camera,
   * because a reader parsing an argument should not have to chase it across the frame.
   *
   * The split exists because uppercase tracked text is measurably slower to read past
   * about five words, so the tag register cannot carry an explanation however long you
   * leave it on screen.
   */
  kind?: 'tag' | 'value' | 'note';
  /** Nudge in CSS pixels, for keeping a label clear of what it points at. */
  dx?: number;
  dy?: number;
}

/** Gap kept between a label's box and the edge of the frame, in CSS pixels. */
const GUTTER = 14;

/**
 * How far a label may be pushed off its anchor to stay in frame, as a fraction of the
 * viewport.
 *
 * Bounded on purpose. A tag names a specific object, so sliding it across the frame
 * would leave it pointing at the wrong thing; the point of the nudge is to rescue a wide
 * label whose anchor is fine, not to keep an off-screen subject labelled. Anything that
 * needs more than this has genuinely left the shot, and `Scene.label` fades it instead.
 */
const NUDGE_X = 0.07;
const NUDGE_Y = 0.05;

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export class AnnotationLayer {
  private pool = new Map<string, HTMLElement>();
  /**
   * Measured label boxes, cached by id.
   *
   * Keeping a label inside the frame needs its width, and reading `offsetWidth` forces a
   * synchronous layout. Doing that for every label every frame would be a dozen forced
   * reflows at 60fps. A label's box only changes when its text or the viewport changes,
   * and both are rare, so it is measured then and remembered.
   */
  private size = new Map<string, { w: number; h: number }>();
  private viewport = { w: 0, h: 0 };

  constructor(private readonly root: HTMLElement) {}

  render(labels: readonly Label[]) {
    const seen = new Set<string>();
    const vw = this.root.clientWidth;
    const vh = this.root.clientHeight;
    if (vw !== this.viewport.w || vh !== this.viewport.h) {
      // Type reflows with the viewport, so every cached box is stale.
      this.viewport = { w: vw, h: vh };
      this.size.clear();
    }

    for (const label of labels) {
      if (label.opacity <= 0.01) continue;
      seen.add(label.id);
      let el = this.pool.get(label.id);
      if (!el) {
        el = document.createElement('div');
        el.className = 'anno';
        this.root.appendChild(el);
        this.pool.set(label.id, el);
      }
      const kind = label.kind ?? 'tag';
      if (el.dataset.kind !== kind) {
        el.dataset.kind = kind;
        el.className = `anno anno--${kind}`;
        this.size.delete(label.id);
      }
      if (el.textContent !== label.text) {
        el.textContent = label.text;
        this.size.delete(label.id);
      }

      let box = this.size.get(label.id);
      if (!box) {
        box = { w: el.offsetWidth, h: el.offsetHeight };
        this.size.set(label.id, box);
      }

      // Where the label wants to be, then the nearest position that keeps its whole box
      // in frame, then a bounded move towards that. Values are pixels against this
      // layer's own box, which is what `left`/`top` resolve against.
      //
      // Not vw/vh units either: on mobile 100vh resolves against the large viewport while
      // the layer is inset:0 on the visual viewport, and with a browser toolbar showing
      // those differ by 50-120px, sliding every label off the point it annotates.
      const wantX = label.x * vw + (label.dx ?? 0);
      const wantY = label.y * vh + (label.dy ?? 0);
      const halfW = box.w / 2;
      const halfH = box.h / 2;
      // A label wider than the frame cannot be satisfied; centre it rather than pinning
      // it to one edge, which at least keeps the middle of the sentence readable.
      const insideX =
        box.w + GUTTER * 2 >= vw ? vw / 2 : clamp(wantX, halfW + GUTTER, vw - halfW - GUTTER);
      const insideY =
        box.h + GUTTER * 2 >= vh ? vh / 2 : clamp(wantY, halfH + GUTTER, vh - halfH - GUTTER);
      const x = wantX + clamp(insideX - wantX, -vw * NUDGE_X, vw * NUDGE_X);
      const y = wantY + clamp(insideY - wantY, -vh * NUDGE_Y, vh * NUDGE_Y);

      // Anchor with left/top, which resolve against this layer, and centre with a
      // transform, which resolves against the label itself. They cannot be combined in
      // one transform because percentages there mean "percent of the label".
      el.style.left = `${x.toFixed(1)}px`;
      el.style.top = `${y.toFixed(1)}px`;
      el.style.transform = 'translate(-50%, -50%)';
      el.style.opacity = label.opacity.toFixed(3);
    }

    for (const [id, el] of this.pool) {
      if (seen.has(id)) continue;
      el.remove();
      this.pool.delete(id);
      this.size.delete(id);
    }
  }

  clear() {
    for (const el of this.pool.values()) el.remove();
    this.pool.clear();
    this.size.clear();
  }
}
