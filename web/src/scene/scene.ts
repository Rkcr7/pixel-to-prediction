/**
 * Turns a `Run` plus a timeline position into a frame.
 *
 * Every value drawn here traces back to a tensor the network actually produced. Nothing
 * is faked for looks, and where something is a visual convention rather than data
 * (spacing, arcs, dust) it does not encode a quantity.
 */

import type { Engine, Run } from '../core/engine';
import { COLORS, hexToRgb } from '../core/palette';
import type { Sampled } from '../core/timeline';
import { clamp01, lerp } from '../core/ease';
import { DEFAULT_POST, Renderer, type CameraState, type PostState } from '../gl/renderer';
import { PARTICLE_STRIDE } from '../gl/shaders/particles';
import { SPRITE_BAR, SPRITE_RING } from '../gl/shaders/sprite';
import type { Label } from '../ui/annotations';
import { C1, C2, CANVAS_RES, CLASSES, HIDDEN, IMG, K1, POOL2 } from './constants';
import {
  arcControl,
  BAR_BASE_Y,
  BAR_MAX_HEIGHT,
  BAR_WIDTH,
  barX,
  candidateSlot,
  CONV1_GRID,
  CANDIDATE_X,
  CONV2_GRID,
  fitDistance,
  POOL2_GRID,
  FOV,
  gridCell,
  HIDDEN_X,
  hiddenSlot,
  kernelSlot,
  mixVec,
  POOL2_BLOCK_X,
  pool2Slot,
  STATIONS,
  Z,
  type Vec3,
} from './layout';

/** World size of a full 28x28 field at the input station. */
const FIELD_SIZE = 4.4;

/** Where the working filter sits while it sweeps: beside its own output, in frame. */
const HERO_KERNEL_SLOT = (z: number): Vec3 => [-2.55, 1.72, z + 0.4];

const POS = hexToRgb(COLORS.positive);
const NEG = hexToRgb(COLORS.negative);
const CHROME = hexToRgb(COLORS.chrome);
const ACCENT = hexToRgb(COLORS.accent);

function absMax(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i]);
    if (v > m) m = v;
  }
  return m || 1;
}

/**
 * Positive and negative extremes, kept apart on purpose.
 *
 * A conv layer's range is not symmetric — conv2 here runs about -8.5 to +5.1 — so
 * normalising both halves by the same number leaves every real activation dim and the
 * plate reads as empty. Each half gets its own scale.
 */
function maxPositive(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m || 1;
}

function maxNegativeMagnitude(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) if (-a[i] > m) m = -a[i];
  return m || 1;
}

export class Scene {
  readonly renderer: Renderer;
  private run: Run | null = null;
  private scales = { conv1: 1, pool1: 1, conv2: 1, pool2: 1 };
  private post: PostState = { ...DEFAULT_POST };
  private camera: CameraState = { eye: [0, 0, 7.2], target: [0, 0, 0], fov: FOV };

  /** Small pointer parallax. Purely decorative, and suppressed while exporting so a
   *  recorded run stays reproducible. */
  parallax: [number, number] = [0, 0];
  parallaxEnabled = true;

  /** Labels anchored to scene points, rebuilt every frame and drawn by the DOM. */
  readonly labels: Label[] = [];

  private label(
    id: string,
    text: string,
    world: Vec3,
    opacity: number,
    options?: { kind?: 'tag' | 'value'; dx?: number; dy?: number },
  ) {
    if (opacity <= 0.012) return;
    const p = this.renderer.project(world);
    if (!p.visible || p.x < -0.25 || p.x > 1.25 || p.y < -0.25 || p.y > 1.25) return;
    this.labels.push({
      id,
      text,
      x: p.x,
      y: p.y,
      opacity,
      kind: options?.kind,
      dx: options?.dx,
      dy: options?.dy,
    });
  }

  constructor(canvas: HTMLCanvasElement, private readonly engine: Engine) {
    this.renderer = new Renderer(canvas);
    this.declareStacks();
    this.renderer.uploadKernels(engine.kernels1, K1, C1);
    this.uploadPrototypes();
    this.buildAmbient();
  }

  private declareStacks() {
    const r = this.renderer;
    r.ensureStack('source', CANVAS_RES, CANVAS_RES, 1);
    r.ensureStack('boxed', 20, 20, 1);
    r.ensureStack('input', IMG, IMG, 1);
    // Same pixels as `input`, rendered cool instead of warm. Used wherever the drawing
    // is a reference layer under something else rather than the subject itself.
    r.ensureStack('ghost', IMG, IMG, 1);
    r.setStackTint('ghost', hexToRgb('#93A8C8'), 0.94);
    r.ensureStack('conv1', IMG, IMG, C1);
    r.ensureStack('pool1', IMG / 2, IMG / 2, C1);
    r.ensureStack('conv2', IMG / 2, IMG / 2, C2);
    r.ensureStack('pool2', POOL2, POOL2, C2);
    r.ensureStack('protos', IMG, IMG, CLASSES);
    r.ensureStack('explain', IMG, IMG, 2);
  }

  private uploadPrototypes() {
    this.renderer.uploadStack('protos', this.engine.prototypes);
  }

  /**
   * No ambient dust.
   *
   * It was drifting through every stage, encoding nothing, and it competed with the
   * signal it was supposed to sit behind. Atmosphere that is not congruent with the data
   * is just noise the eye has to filter, and here it actively hurt legibility. The ground
   * glow and the bloom carry the depth on their own.
   */
  private buildAmbient() {
    this.renderer.createParticleGroup('ambient', 1, { colorMode: 0 });
    this.renderer.setParticleData('ambient', new Float32Array(PARTICLE_STRIDE), 0);
  }

  /**
   * Forget the current run. Without this, returning to the drawing screen leaves the
   * previous run's final frame painted behind the compose panel.
   */
  clearRun() {
    this.run = null;
    this.renderer.setParticleState('flowA', 0, 1);
    this.renderer.setParticleState('flowB', 0, 1);
  }

  /** Upload everything for a new run and rebuild its particle streams. */
  setRun(run: Run, source: Float32Array) {
    this.run = run;
    const r = this.renderer;

    r.uploadStack('source', source);
    r.uploadStack('input', run.input);
    r.uploadStack('ghost', run.input);

    // Pad the aspect-preserved box into a fixed 20x20 plane so it has a stable texture.
    const boxed = new Float32Array(400);
    const ox = Math.floor((20 - run.prep.boxedW) / 2);
    const oy = Math.floor((20 - run.prep.boxedH) / 2);
    for (let y = 0; y < run.prep.boxedH; y++) {
      for (let x = 0; x < run.prep.boxedW; x++) {
        boxed[(oy + y) * 20 + ox + x] = run.prep.boxed[y * run.prep.boxedW + x];
      }
    }
    r.uploadStack('boxed', boxed);

    // conv1 is uploaded *pre*-activation so the shader can extinguish the negative half
    // on cue. The positive half is identical either way, which is the point.
    r.uploadStack('conv1', run.conv1Pre);
    r.uploadStack('pool1', run.pool1);
    // conv2 ships post-ReLU: stage 4 is about what the next layer receives, and ReLU is
    // already the lesson of stage 3. Showing the pre-activation here would leave 84% of
    // every plate at zero with nothing to say about it.
    r.uploadStack('conv2', run.conv2);
    r.uploadStack('pool2', run.pool2);

    const explain = new Float32Array(784 * 2);
    explain.set(run.saliency, 0);
    explain.set(run.counterfactual, 784);
    r.uploadStack('explain', explain);

    // One normalisation per layer, not per channel, so channels stay comparable to
    // each other — a dim feature map should look dim.
    this.scales = {
      conv1: 1 / maxPositive(run.conv1Pre),
      pool1: 1 / maxPositive(run.pool1),
      conv2: 1 / maxPositive(run.conv2),
      pool2: 1 / maxPositive(run.pool2),
    };
    r.setStackNegScale('conv1', 1 / maxNegativeMagnitude(run.conv1Pre));
    r.setStackNegScale('conv2', 1);
    r.setStackNegScale('input', 1);

    this.buildFlowA(run);
    this.buildFlowB(run);
  }

  /**
   * pool2 cells to hidden units, along the strongest contributing edges.
   *
   * Only the top few edges per unit get particles. Drawing all 25,088 connections, or
   * even the top ten per unit, produces an additive glow soup where nothing is
   * distinguishable — the count has to stay low enough that a single stream can be
   * followed with the eye.
   */
  private buildFlowA(run: Run) {
    const triples = run.fc1TopContributions;
    const allEdges = Math.floor(triples.length / 3);
    const PER_UNIT = 3;
    const kept: number[] = [];
    const seenPerUnit = new Map<number, number>();
    for (let e = 0; e < allEdges; e++) {
      const dst = triples[e * 3 + 1];
      const used = seenPerUnit.get(dst) ?? 0;
      if (used >= PER_UNIT) continue;
      seenPerUnit.set(dst, used + 1);
      kept.push(e);
    }

    const perEdge = 1;
    const count = kept.length * perEdge;
    const data = new Float32Array(count * PARTICLE_STRIDE);
    const scale = absMax(triples.filter((_, i) => i % 3 === 2)) || 1;

    let n = 0;
    for (const e of kept) {
      const src = triples[e * 3];
      const dst = triples[e * 3 + 1];
      const value = triples[e * 3 + 2] / scale;
      const channel = Math.floor(src / (POOL2 * POOL2));
      const within = src % (POOL2 * POOL2);
      const cx = within % POOL2;
      const cy = Math.floor(within / POOL2);

      const base = pool2Slot(channel, Z.dense);
      const from: Vec3 = [
        base[0] + (cx / (POOL2 - 1) - 0.5) * 0.72,
        base[1] - (cy / (POOL2 - 1) - 0.5) * 0.72,
        base[2],
      ];
      const to = hiddenSlot(dst, Z.dense);
      const ctrl = arcControl(from, to, 0.55, e);

      for (let k = 0; k < perEdge; k++) {
        writeParticle(data, n++, from, to, ctrl, {
          seed: ((e * 7 + k * 13) % 97) / 97,
          speed: 0.2 + (k % 3) * 0.025,
          size: 0.075 + Math.abs(value) * 0.06,
          value: 0.55 + Math.abs(value) * 0.45,
          opacity: 0.4 + Math.abs(value) * 0.5,
        });
      }
    }
    this.renderer.createParticleGroup('flowA', Math.max(count, 1), { colorMode: 0 });
    this.renderer.setParticleData('flowA', data, n);
  }

  /**
   * Hidden units to the ten candidates, coloured by sign.
   *
   * This is the stream that shows *inhibition*: coral particles arriving at a candidate
   * are evidence against it. Almost no visualisation shows subtraction, and it is the
   * most counter-intuitive thing the network does.
   */
  private buildFlowB(run: Run) {
    const contrib = run.fc2Contributions; // [class * 32 + unit]

    // Per class, the two strongest arguments for and the strongest against. Ranking the
    // whole 320-edge matrix globally lets one loud class own every particle, and drawing
    // 150 edges at once is a soup; three legible streams per candidate says the same
    // thing and can actually be followed.
    const keep: number[] = [];
    for (let c = 0; c < CLASSES; c++) {
      const row = Array.from({ length: HIDDEN }, (_, u) => c * HIDDEN + u);
      const positive = row
        .filter((i) => contrib[i] > 0)
        .sort((a, b) => contrib[b] - contrib[a])
        .slice(0, 2);
      const negative = row
        .filter((i) => contrib[i] < 0)
        .sort((a, b) => contrib[a] - contrib[b])
        .slice(0, 1);
      keep.push(...positive, ...negative);
    }

    const scale = absMax(contrib);
    const perEdge = 2;
    const data = new Float32Array(Math.max(keep.length, 1) * perEdge * PARTICLE_STRIDE);

    let n = 0;
    keep.forEach((idx, e) => {
      const cls = Math.floor(idx / HIDDEN);
      const unit = idx % HIDDEN;
      const value = contrib[idx] / scale;
      const from = hiddenSlot(unit, Z.dense);
      const to = candidateSlot(cls, Z.dense);
      const ctrl = arcControl(from, to, 0.55, e);
      for (let k = 0; k < perEdge; k++) {
        writeParticle(data, n++, from, to, ctrl, {
          seed: ((e * 11 + k * 17) % 53) / 53,
          speed: 0.17 + (k % 2) * 0.03,
          size: 0.085 + Math.abs(value) * 0.075,
          value,
          opacity: 0.5 + Math.abs(value) * 0.5,
        });
      }
    });
    this.renderer.createParticleGroup('flowB', Math.max(keep.length * perEdge, 1), {
      colorMode: 1,
    });
    this.renderer.setParticleData('flowB', data, n);
  }

  // -- per-frame -------------------------------------------------------------

  draw(v: Sampled, time: number, frameMs: number) {
    const run = this.run;
    const r = this.renderer;
    r.beginFrame();
    this.labels.length = 0;

    this.updateCamera(v);
    r.setCamera(this.camera);
    this.updatePost(v);

    if (run) {
      this.drawInputStation(v, run, time);
      this.drawConvStation(v, run);
      this.drawShapeStation(v, run);
      this.drawDenseStation(v, run);
      this.drawDecisionStation(v, run);
      this.drawAnswerStation(v, run, time);
    } else {
      // No run: draw nothing, but still render. Returning early here skips the clear,
      // and WebGL keeps presenting the last composited frame — which is why the previous
      // answer stayed on screen behind the drawing panel.
      this.post.groundGlow = 0.16;
      this.post.aberration = 0;
      this.post.defocus = 0;
      this.post.fade = 1;
    }

    r.setParticleState('flowA', run ? (v['s5.flowA'] ?? 0) : 0, 1);
    r.setParticleState('flowB', run ? (v['s5.flowB'] ?? 0) : 0, 1);

    r.render(this.post, time, frameMs);
  }

  private updateCamera(v: Sampled) {
    const s = clamp01((v['cam.station'] ?? 0) / (STATIONS.length - 1)) * (STATIONS.length - 1);
    const i = Math.min(STATIONS.length - 2, Math.floor(s));
    const f = s - i;
    const z = lerp(STATIONS[i].z, STATIONS[i + 1].z, f);

    // Distance is solved per frame from the content bounds and the live aspect, so the
    // framing survives any window shape including portrait phones. `cam.zoom` then lets
    // a beat push in on a single element — a filter reading the image, or one map being
    // pooled — which is the difference between watching a process and squinting at it.
    const aspect = this.renderer.aspect;
    const zoom = v['cam.zoom'] ?? 1;
    const dist =
      lerp(fitDistance(STATIONS[i], aspect), fitDistance(STATIONS[i + 1], aspect), f) * zoom;

    // A station's centre is chosen to frame all of its content. When a beat pushes in on
    // one element the rest of that content is no longer in shot, so the aim has to come
    // back down with the zoom or the subject falls out of the bottom of the frustum.
    const cy =
      lerp(STATIONS[i].centerY, STATIONS[i + 1].centerY, f) + (v['cam.centerAdjust'] ?? 0);
    const px = this.parallaxEnabled ? this.parallax[0] * 0.4 : 0;
    const py = this.parallaxEnabled ? this.parallax[1] * 0.26 : 0;
    this.camera.eye = [px, cy + py, z + dist];
    this.camera.target = [px * 0.35, cy + py * 0.25, z];
    this.camera.fov = FOV;
  }

  private updatePost(v: Sampled) {
    const p = this.post;
    p.fade = v['post.fade'] ?? 1;
    p.aberration = v['post.aberration'] ?? 0;
    p.defocus = v['post.defocus'] ?? 0;
    p.groundGlow = v['post.groundGlow'] ?? 0.34;
    p.exposure = v['post.exposure'] ?? 1;
    p.bloomWeight = DEFAULT_POST.bloomWeight;
    p.vignette = DEFAULT_POST.vignette;
    p.grain = DEFAULT_POST.grain;
  }

  // Stage 1 and 2 -------------------------------------------------------------
  private drawInputStation(v: Sampled, run: Run, time: number) {
    const r = this.renderer;
    const z = Z.input;
    const scale = v['s1.scale'] ?? 1;
    // A slow breath. Derived from timeline time, so it is still a pure function of t.
    const breathe = 1 + Math.sin(time * 1.05) * 0.006;

    const sourceOpacity = (v['s1.digit'] ?? 0) * (v['s2.source'] ?? 1);
    if (sourceOpacity > 0.002) {
      const s = FIELD_SIZE * scale * breathe;
      r.plate('source', 0, [0, 0, z], [s, s], {
        opacity: sourceOpacity,
        valueScale: 1,
        cellBias: 0,
      });
    }

    // The crop frame: where the ink actually is.
    const ring = v['s2.cropRing'] ?? 0;
    if (ring > 0.002) {
      const [bx, by, bw, bh] = run.prep.bbox;
      const cx = ((bx + bw / 2) / CANVAS_RES - 0.5) * FIELD_SIZE;
      const cy = -((by + bh / 2) / CANVAS_RES - 0.5) * FIELD_SIZE;
      const w = (bw / CANVAS_RES) * FIELD_SIZE;
      const h = (bh / CANVAS_RES) * FIELD_SIZE;
      r.sprite([cx, cy, z + 0.02], [w + 0.12, h + 0.12], ACCENT, ring, {
        mode: SPRITE_RING,
        radius: 0.06,
        softness: 0.018,
        intensity: 1.6,
      });
    }

    // The 20-pixel box, then its translation onto the centre of mass.
    const boxOpacity = v['s2.boxed'] ?? 0;
    if (boxOpacity > 0.002) {
      const px = FIELD_SIZE / IMG;
      const w = px * 20;
      const cx28 = run.prep.offset[0] + run.prep.boxedW / 2;
      const cy28 = run.prep.offset[1] + run.prep.boxedH / 2;
      const targetX = (cx28 - IMG / 2) * px;
      const targetY = -(cy28 - IMG / 2) * px;
      const centre = v['s2.centre'] ?? 0;
      const plateX = targetX * centre;
      const plateY = targetY * centre;

      r.plate('boxed', 0, [plateX, plateY, z + 0.01], [w, w], {
        opacity: boxOpacity,
        valueScale: 1,
        cellBias: 0.5,
      });

      const dot = v['s2.comDot'] ?? 0;
      if (dot > 0.002) {
        // The marker has to sit on the ink's actual centre of mass, offset from the
        // middle of its own box — that offset is precisely what the translation is
        // about to cancel. Pinning it to the plate centre made the beat meaningless,
        // because the dot was already where it was supposed to end up.
        const comX = (run.prep.com[0] - run.prep.boxedW / 2) * px;
        const comY = -(run.prep.com[1] - run.prep.boxedH / 2) * px;

        r.sprite([plateX + comX, plateY + comY, z + 0.06], [0.17, 0.17], ACCENT, dot, {
          intensity: 2.6,
          softness: 0.045,
        });
        // Where it is heading: the centre of the 28x28 field.
        r.sprite([0, 0, z + 0.05], [0.52, 0.52], CHROME, dot * 0.55, {
          mode: SPRITE_RING,
          softness: 0.012,
          intensity: 1.1,
        });
        this.label(
          's2.com',
          'centre of mass',
          [plateX + comX, plateY + comY - 0.55, z + 0.06],
          dot * 0.9,
          { kind: 'tag' },
        );
      }
    }

    const inputOpacity = v['s2.input'] ?? 0;
    if (inputOpacity > 0.002) {
      r.plate('input', 0, [0, 0, z + 0.02], [FIELD_SIZE, FIELD_SIZE], {
        opacity: inputOpacity,
        valueScale: 1,
        cellBias: 1,
        grid: v['s2.grid'] ?? 0,
      });
    }
  }

  // Stage 3 -------------------------------------------------------------------
  private drawConvStation(v: Sampled, run: Run) {
    const r = this.renderer;
    const z = Z.conv1;
    const fade = v['s3.fade'] ?? 1;
    if (fade <= 0.002) return;

    const spread = v['s3.spread'] ?? 0;
    const hero = run.conv1Order[0];
    const contract = v['s4.contract'] ?? 0;
    const relu = v['s3.relu'] ?? 0;
    const signed = v['s3.signed'] ?? 0;
    const negBoost = v['s3.reluHint'] ?? 0;
    // Stage 4 opens by coming back to one map and looking at it closely, because
    // pooling is invisible at grid scale: 2x2 windows on a 2.5-unit plate are a
    // couple of pixels each.
    const focus = v['s4.focus'] ?? 0;
    const contest = v['s4.contest'] ?? 0;

    // The learned kernels, above the plates.
    const lift = v['s3.heroLift'] ?? 0;
    for (let i = 0; i < C1; i++) {
      const o = (v[`s3.kernel${i}`] ?? 0) * fade;
      if (o <= 0.002) continue;
      const isHero = i === hero;
      const heroBeat = (v['s3.heroKernel'] ?? 0) * (1 - spread);
      const highlight = isHero ? heroBeat * 0.9 : 0;
      // Dim the others while one filter is doing the work, so it is obvious which
      // kernel produced the response being drawn.
      const dim = isHero ? 1 : lerp(1, 0.28, heroBeat);

      // The working filter travels down beside the plate it is writing. The camera
      // pushes in for the sweep and would otherwise crop the kernel row entirely,
      // severing the only visible link between a filter and its own output.
      const rowSlot = kernelSlot(i, C1, z);
      const pos = isHero ? mixVec(rowSlot, HERO_KERNEL_SLOT(z), lift) : rowSlot;
      const size = isHero ? lerp(0.86, 1.32, lift) : 0.86;
      // Stage 4 is about pooling, and the filter row sits above the frame it needs.
      // Clear it out so the close-up has the shot to itself.
      r.kernel(i, pos, size, o * dim * (1 - focus), 3.4, highlight);

      if (isHero && lift > 0.05) {
        this.label('s3.thisFilter', 'this filter', [pos[0], pos[1] - 1.0, pos[2]], lift * fade * 0.9, {
          kind: 'tag',
        });
      }
    }

    const kernelsIn = (v['s3.kernel0'] ?? 0) * fade;
    this.label('s3.kernels', 'the 8 filters it learned', [0, 4.18, z], kernelsIn * 0.9, {
      kind: 'tag',
    });
    // Name the hero filter so the sweep is clearly one specific filter, not "the network".
    this.label(
      's3.hero',
      `filter ${hero + 1} reading the image`,
      [0, -2.55, z],
      (v['s3.hero'] ?? 0) * (1 - spread) * fade,
      { kind: 'tag' },
    );
    this.label(
      's3.relu',
      'anything below zero becomes zero',
      [0, -3.5, z],
      Math.max(v['s3.reluHint'] ?? 0, relu) * fade * 0.95,
      { kind: 'tag' },
    );

    const heroCenter: Vec3 = [0, 0, z];
    for (let i = 0; i < C1; i++) {
      const staged = v[`s3.plate${i}`] ?? 0;
      const heroOn = i === hero ? (v['s3.hero'] ?? 0) : 0;
      const opacity = Math.max(staged, heroOn) * fade;
      if (opacity <= 0.002) continue;

      const isHero = i === hero;
      const grid = gridCell(CONV1_GRID, i, z);
      // Spread out into the grid, then the hero comes back to the middle for the
      // pooling close-up while the rest recede.
      const target = isHero ? mixVec(grid, heroCenter, focus) : grid;
      const pos = mixVec(heroCenter, target, Math.max(spread, focus));
      const size =
        lerp(4.5, CONV1_GRID.plate, spread) * (isHero ? lerp(1, 4.7 / CONV1_GRID.plate, focus) : 1);
      const sweep = isHero ? (v['s3.sweep'] ?? 1) : spread > 0 ? 1 : 0;
      const recede = isHero ? 1 : lerp(1, 0.04, focus);

      // Cross-fade to the pooled field: the same map, half the cells. Showing them at
      // the same physical size makes the cell coarsening the only thing that changed,
      // which is exactly what pooling did.
      r.plate('conv1', i, pos, [size, size], {
        opacity: opacity * (1 - contract) * recede,
        valueScale: this.scales.conv1,
        cellBias: 0.85,
        grid: isHero ? focus * 0.55 : 0,
      }, {
        signedMix: signed * (1 + negBoost * 0.8),
        sweep,
        highlight: isHero ? heroOn * 0.25 : 0,
        dissolve: relu,
      });

      if (contract > 0.002) {
        r.plate('pool1', i, pos, [size, size], {
          opacity: opacity * contract * recede,
          valueScale: this.scales.pool1,
          cellBias: 1,
          grid: isHero ? focus * 0.8 : 0,
        }, { highlight: contest * 0.7 });
      }
    }

    this.label(
      's4.pooling',
      'each 2 by 2 block keeps only its brightest cell',
      [0, -2.9, z],
      focus * fade * 0.95,
      { kind: 'tag' },
    );
    this.label(
      's4.halved',
      `${IMG} across becomes ${IMG / 2}`,
      [0, 2.85, z],
      contract * focus * fade * 0.9,
      { kind: 'tag' },
    );
  }

  // Stage 4 -------------------------------------------------------------------
  private drawShapeStation(v: Sampled, run: Run) {
    const r = this.renderer;
    const z = Z.conv2;
    const fade = v['s4.fade'] ?? 1;
    if (fade <= 0.002) return;

    const rank = v['s4.rank'] ?? 0;
    const pool = v['s4.pool'] ?? 0;
    const energies = run.conv2Energy;
    const maxEnergy = Math.max(...Array.from(energies)) || 1;

    for (let i = 0; i < C2; i++) {
      const opacity = (v[`s4.plate${i}`] ?? 0) * fade;
      if (opacity <= 0.002) continue;
      const pos = gridCell(CONV2_GRID, i, z);
      const size = CONV2_GRID.plate * lerp(1, 0.78, pool);

      // Selective luminance: strong maps hold, weak maps recede. The ranking is real —
      // it is the sum of squares of each map's activation. The floor stays well above
      // zero so a quiet map still reads as a map rather than disappearing.
      const strength = energies[i] / maxEnergy;
      const dim = lerp(1, 0.48 + strength * 0.62, rank);
      const highlight = rank * Math.max(0, strength - 0.55) * 1.1;

      // A ghost of the drawing under every feature map.
      //
      // A 14x14 map that is 84% zeros is a few scattered blobs, and blobs with no frame
      // of reference mean nothing. Sitting them over the digit turns each one into a
      // readable statement: this unit fires on the loop, that one on the tail.
      // Faint enough to be a reference frame, not the subject. Too strong and every
      // tile just reads as "the digit again" instead of "where this feature fires".
      r.plate('ghost', 0, [pos[0], pos[1], pos[2] - 0.05], [size, size], {
        opacity: opacity * 0.32 * dim,
        valueScale: 1,
        cellBias: 0,
      });

      r.plate('conv2', i, pos, [size, size], {
        opacity: opacity * dim * (1 - pool),
        valueScale: this.scales.conv2,
        cellBias: 0.9,
      }, { signedMix: 0, highlight });

      if (pool > 0.002) {
        r.plate('pool2', i, pos, [size, size], {
          opacity: opacity * dim * pool,
          valueScale: this.scales.pool2,
          cellBias: 1,
        }, { highlight });
      }
    }

    const arrived = (v[`s4.plate${run.conv2Order[0]}`] ?? 0) * fade;
    this.label(
      's4.combined',
      '16 new features, each one mixing all 8 edges',
      [0, 4.55, z],
      arrived * 0.92,
      { kind: 'tag' },
    );
    // Name the winner rather than leaving the ranking implicit.
    const top = run.conv2Order[0];
    this.label(
      's4.top',
      'strongest response',
      [gridCell(CONV2_GRID, top, z)[0], gridCell(CONV2_GRID, top, z)[1] - 1.12, z],
      rank * fade * 0.95,
      { kind: 'tag' },
    );
  }

  // Stage 5 -------------------------------------------------------------------
  private drawDenseStation(v: Sampled, run: Run) {
    const r = this.renderer;
    const z = Z.dense;
    const block = v['s5.block'] ?? 0;
    const gather = v['s6.gather'] ?? 0;
    const decisionFade = v['s6.fade'] ?? 1;

    if (block > 0.002 && gather < 0.999) {
      for (let i = 0; i < C2; i++) {
        r.plate('pool2', i, pool2Slot(i, z), [POOL2_GRID.plate, POOL2_GRID.plate], {
          opacity: block * (1 - gather) * decisionFade,
          valueScale: this.scales.pool2,
          cellBias: 1,
          grid: 0.35,
        });
      }
    }

    // The 32 hidden units.
    //
    // ReLU zeroes roughly half of them, so sizing purely by activation makes half the
    // layer vanish and the structure with it. Every unit keeps a faint socket so the
    // lattice stays readable; the filled core carries the value.
    const fc1Max = Math.max(...Array.from(run.fc1)) || 1;
    for (let i = 0; i < HIDDEN; i++) {
      const o = (v[`s5.unit${i}`] ?? 0) * (1 - gather) * decisionFade;
      if (o <= 0.002) continue;
      const slot = hiddenSlot(i, z);
      const a = run.fc1[i] / fc1Max;

      r.sprite(slot, [0.27, 0.27], CHROME, o * 0.3, {
        mode: SPRITE_RING,
        softness: 0.012,
        intensity: 0.85,
      });
      if (a > 0.002) {
        const size = 0.07 + a * 0.17;
        r.sprite(slot, [size, size], POS, o * (0.42 + a * 0.58), {
          intensity: 1.1 + a * 3.0,
          softness: 0.035,
        });
      }
    }

    // The ten candidates: the network's own average of each digit, not a font.
    const contrib = run.fc2Contributions;
    const weigh = v['s5.weigh'] ?? 0;
    const positiveMass = new Float32Array(CLASSES);
    const negativeMass = new Float32Array(CLASSES);
    for (let c = 0; c < CLASSES; c++) {
      let up = 0;
      let down = 0;
      for (let u = 0; u < HIDDEN; u++) {
        const w = contrib[c * HIDDEN + u];
        if (w > 0) up += w;
        else down -= w;
      }
      positiveMass[c] = up;
      negativeMass[c] = down;
    }
    const massMax = Math.max(...Array.from(positiveMass)) || 1;
    const oppositionMax = Math.max(massMax, ...Array.from(negativeMass)) || 1;

    for (let c = 0; c < CLASSES; c++) {
      const o = (v[`s5.cand${c}`] ?? 0) * decisionFade;
      if (o <= 0.002) continue;
      const home = candidateSlot(c, z);
      // Clear of the container so the percentage label has somewhere to sit.
      const barTarget: Vec3 = [barX(c), BAR_BASE_Y + BAR_MAX_HEIGHT + 1.0, z];
      const pos = mixVec(home, barTarget, gather);
      const mass = positiveMass[c] / massMax;
      // Once the answer locks, the winning candidate is the only one still fully lit.
      const winner = c === run.ranked[0];
      const lockAmount = v['s6.lock'] ?? 0;
      const lit =
        lerp(0.45, 0.3 + mass * 1.5, weigh) * lerp(1, winner ? 1.5 : 0.28, lockAmount);
      const size = lerp(0.56, 0.44, gather);
      r.plate('protos', c, pos, [size, size], {
        opacity: o * lit,
        valueScale: 1,
        cellBias: 0.2,
      });

      // A tug of war beside each candidate.
      //
      // Back to back from a shared centre line: coral to the left is evidence against,
      // cyan to the right is evidence for, and the bright tick marks the net. The net is
      // what actually decides, and without it the meter can mislead — a class with huge
      // support and huge opposition looks busy but nets out to nothing.
      const meter = o * (1 - gather);
      if (meter > 0.01 && weigh > 0.004) {
        // Grow outward from the axis rather than fading in at full length. A bar whose
        // length is final from the first frame reads as an instant pop however long the
        // opacity fade is.
        const unit = (1.05 / oppositionMax) * weigh;
        const forLen = positiveMass[c] * unit;
        const againstLen = negativeMass[c] * unit;
        const axis = pos[0] - 1.42;

        r.sprite([axis, pos[1], pos[2] - 0.02], [0.014, 0.3], CHROME, meter * 0.5, {
          mode: SPRITE_BAR,
          radius: 0.006,
          softness: 0.008,
          intensity: 0.9,
        });
        r.sprite(
          [axis + forLen / 2, pos[1], pos[2]],
          [Math.max(forLen, 0.02), 0.07],
          POS,
          meter * 0.95,
          { mode: SPRITE_BAR, radius: 0.028, softness: 0.018, intensity: 1.5 },
        );
        r.sprite(
          [axis - againstLen / 2, pos[1], pos[2]],
          [Math.max(againstLen, 0.02), 0.07],
          NEG,
          meter * 0.85,
          { mode: SPRITE_BAR, radius: 0.028, softness: 0.018, intensity: 1.3 },
        );

        const net = (positiveMass[c] - negativeMass[c]) * unit;
        r.sprite([axis + net, pos[1], pos[2] + 0.04], [0.055, 0.24], ACCENT, meter, {
          mode: SPRITE_BAR,
          radius: 0.02,
          softness: 0.014,
          intensity: 2.2,
        });
      }
    }

    // Name the three groups. Without this the middle of the network is just dots.
    const showGroups = (1 - gather) * decisionFade;
    this.label('s5.features', '784 surviving features', [POOL2_BLOCK_X, 2.35, z], block * showGroups * 0.9, {
      kind: 'tag',
    });
    this.label('s5.units', '32 hidden units', [HIDDEN_X, 2.72, z], (v['s5.unit0'] ?? 0) * showGroups * 0.9, {
      kind: 'tag',
    });
    this.label('s5.cands', '10 candidates', [CANDIDATE_X, 3.62, z], (v['s5.cand0'] ?? 0) * showGroups * 0.9, {
      kind: 'tag',
    });
    this.label(
      's5.sign',
      'cyan argues for, coral argues against',
      [HIDDEN_X + 1.6, -3.15, z],
      (v['s5.flowB'] ?? 0) * showGroups * 0.9,
      { kind: 'tag' },
    );
  }

  // Stage 6 -------------------------------------------------------------------
  private drawDecisionStation(v: Sampled, run: Run) {
    const r = this.renderer;
    const z = Z.dense;
    const show = v['s6.logits'] ?? 0;
    const fade = v['s6.fade'] ?? 1;
    if (show <= 0.002 || fade <= 0.002) return;

    const exp = v['s6.exp'] ?? 0;
    const norm = v['s6.normalize'] ?? 0;
    const lockAmount = v['s6.lock'] ?? 0;
    const winner = run.ranked[0];

    const logitMax = absMax(run.logits);
    const expMax = Math.max(...Array.from(run.exps)) || 1;

    // The budget: one unit of certainty, drawn as a container that the shares fill.
    const budget = v['s6.budget'] ?? 0;
    if (budget > 0.002) {
      r.sprite(
        [0, BAR_BASE_Y + BAR_MAX_HEIGHT * 0.5, z - 0.05],
        [CLASSES * 1.02 + 0.35, BAR_MAX_HEIGHT + 0.3],
        CHROME,
        budget * 0.32 * fade,
        { mode: SPRITE_RING, radius: 0.12, softness: 0.014, intensity: 0.8 },
      );
    }

    // The zero line. Raw scores can be negative, and hiding that would be a lie.
    r.sprite([0, BAR_BASE_Y, z - 0.04], [CLASSES * 1.02 + 0.2, 0.012], CHROME, show * 0.5 * fade, {
      mode: SPRITE_BAR,
      radius: 0.006,
      softness: 0.006,
      intensity: 0.9,
    });

    for (let c = 0; c < CLASSES; c++) {
      // Three successive readings of the same number: raw score, exponentiated,
      // normalised. Each is the real value at that point in the computation.
      const raw = (run.logits[c] / logitMax) * (BAR_MAX_HEIGHT * 0.42);
      const expH = (run.exps[c] / expMax) * (BAR_MAX_HEIGHT * 0.86);
      const probH = run.probs[c] * BAR_MAX_HEIGHT;

      let h = lerp(raw, expH, exp);
      h = lerp(h, probH, norm);

      const isWinner = c === winner;
      const emphasise = isWinner ? lockAmount : 0;
      const opacity = show * fade * lerp(1, isWinner ? 1 : 0.32, lockAmount);

      const colour = isWinner ? ACCENT : h >= 0 ? POS : NEG;
      const height = Math.abs(h) + 0.02;
      const centreY = BAR_BASE_Y + h / 2;

      r.sprite(
        [barX(c), centreY, z],
        [BAR_WIDTH * lerp(1, 1.18, emphasise), height],
        colour,
        opacity,
        {
          mode: SPRITE_BAR,
          radius: 0.09,
          softness: 0.03,
          intensity: 1.0 + Math.abs(h / BAR_MAX_HEIGHT) * 2.2 + emphasise * 1.8,
        },
      );

      // Which digit this bar is. The ghost prototypes above are evocative but not
      // unambiguous, and at this point in the story precision matters more than mood.
      this.label(`s6.d${c}`, String(c), [barX(c), BAR_BASE_Y - 0.34, z], show * fade * 0.85, {
        kind: 'value',
      });

      // The number itself, in whichever reading is currently on screen. During the
      // exponential beat there is no meaningful unit to quote, so it stays quiet and
      // lets the bars carry it.
      let text = '';
      let strength = 0;
      if (norm > 0.35) {
        const pct = run.probs[c] * 100;
        text = pct >= 0.05 ? `${pct.toFixed(pct >= 99.95 ? 1 : 0)}%` : '';
        strength = norm;
      } else if (exp < 0.3) {
        text = run.logits[c].toFixed(1);
        strength = show * (1 - exp);
      }
      if (text) {
        this.label(
          `s6.v${c}`,
          text,
          [barX(c), BAR_BASE_Y + h + (h >= 0 ? 0.34 : -0.34), z],
          strength * fade * (isWinner ? 1 : 0.72),
          { kind: 'value' },
        );
      }
    }

    // Name the reading. Softmax is two operations and calling them out as they happen is
    // the difference between "the bars moved" and "the gaps were stretched, then the
    // total was split".
    this.label('s6.readRaw', 'raw score', [0, BAR_MAX_HEIGHT + 0.4, z], show * (1 - exp) * fade * 0.9, {
      kind: 'tag',
    });
    this.label(
      's6.readExp',
      'exponentiate: the gaps stretch',
      [0, BAR_MAX_HEIGHT + 0.4, z],
      exp * (1 - norm) * fade * 0.9,
      { kind: 'tag' },
    );
    this.label(
      's6.readNorm',
      'one unit of certainty, split ten ways',
      [0, BAR_MAX_HEIGHT + 0.4, z],
      norm * fade * 0.9,
      { kind: 'tag' },
    );
  }

  // Stage 7 -------------------------------------------------------------------
  private drawAnswerStation(v: Sampled, run: Run, time: number) {
    const r = this.renderer;
    const z = Z.answer;
    const digit = v['s7.digit'] ?? 0;
    if (digit <= 0.002) return;

    const size = FIELD_SIZE * 0.95;

    // The network's own average of the winning digit, sitting faintly behind the
    // drawing. Seeing your stroke against what the model expects that digit to look
    // like is the moment the whole thing clicks for most people.
    r.plate('protos', run.ranked[0], [0, 0, z - 0.06], [size * 1.02, size * 1.02], {
      opacity: digit * 0.16,
      valueScale: 1,
      cellBias: 0,
    });

    // Your ink, cool and quiet. The warm attribution layer then reads as a separate
    // statement on top of it rather than merging into one glowing digit.
    r.plate('ghost', 0, [0, 0, z], [size, size], {
      opacity: digit * 0.42,
      valueScale: 1,
      cellBias: 0.35,
    });

    // Which ink produced the answer. Real gradients, not a decorative glow.
    const sal = v['s7.saliency'] ?? 0;
    if (sal > 0.002) {
      r.plate('explain', 0, [0, 0, z + 0.02], [size, size], {
        opacity: sal * 1.25,
        valueScale: 1,
        cellBias: 0.25,
      });
    }

    // What nearly changed its mind, pulsing gently so it reads as a separate layer.
    const counter = v['s7.counter'] ?? 0;
    if (counter > 0.002) {
      const pulse = 0.72 + Math.sin(time * 2.1) * 0.28;
      r.plate('explain', 1, [0, 0, z + 0.04], [size, size], {
        opacity: counter * pulse * 0.8,
        valueScale: 1,
        cellBias: 0.25,
      }, { signedMix: 1 });
    }
  }
}

interface ParticleOptions {
  seed: number;
  speed: number;
  size: number;
  value: number;
  opacity: number;
}

function writeParticle(
  data: Float32Array,
  index: number,
  from: Vec3,
  to: Vec3,
  ctrl: Vec3,
  o: ParticleOptions,
) {
  const i = index * PARTICLE_STRIDE;
  if (i + PARTICLE_STRIDE > data.length) return;
  data[i] = from[0];
  data[i + 1] = from[1];
  data[i + 2] = from[2];
  data[i + 3] = to[0];
  data[i + 4] = to[1];
  data[i + 5] = to[2];
  data[i + 6] = ctrl[0];
  data[i + 7] = ctrl[1];
  data[i + 8] = ctrl[2];
  data[i + 9] = o.seed;
  data[i + 10] = o.speed;
  data[i + 11] = o.size;
  data[i + 12] = o.value;
  data[i + 13] = o.opacity;
  data[i + 14] = 0; // flow mode
  data[i + 15] = 0;
}
