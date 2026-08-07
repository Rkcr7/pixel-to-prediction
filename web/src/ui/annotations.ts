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

export class AnnotationLayer {
  private pool = new Map<string, HTMLElement>();

  constructor(private readonly root: HTMLElement) {}

  render(labels: readonly Label[]) {
    const seen = new Set<string>();

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
      }
      if (el.textContent !== label.text) el.textContent = label.text;
      // Anchor with left/top, which resolve against this layer, and centre with a
      // transform, which resolves against the label itself. They cannot be combined in
      // one transform because percentages there mean "percent of the label".
      //
      // Not vw/vh either: on mobile 100vh resolves against the large viewport while the
      // layer is inset:0 on the visual viewport, and with a browser toolbar showing
      // those differ by 50-120px, sliding every label off the point it annotates.
      el.style.left = `${(label.x * 100).toFixed(3)}%`;
      el.style.top = `${(label.y * 100).toFixed(3)}%`;
      el.style.transform = `translate(calc(-50% + ${label.dx ?? 0}px), calc(-50% + ${
        label.dy ?? 0
      }px))`;
      el.style.opacity = label.opacity.toFixed(3);
    }

    for (const [id, el] of this.pool) {
      if (seen.has(id)) continue;
      el.remove();
      this.pool.delete(id);
    }
  }

  clear() {
    for (const el of this.pool.values()) el.remove();
    this.pool.clear();
  }
}
