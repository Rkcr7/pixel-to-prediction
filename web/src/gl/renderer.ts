/**
 * The renderer. Owns the GL resources and knows nothing about neural networks — stages
 * hand it plates, kernels and particle streams, and it draws them.
 *
 * Frame shape: one HDR scene pass (everything additive, depth off), a quarter-resolution
 * bloom pyramid, then a single composite pass to the default framebuffer.
 */

import {
  createContext,
  createFieldTexture,
  FieldStack,
  FullscreenTriangle,
  Program,
  RenderTarget,
  uploadField,
  VERT_FULLSCREEN,
  type GLContext,
} from './gl';
import { lookAt, multiply, perspective, type Mat4 } from './mat4';
import { KERNEL_FRAG, KERNEL_VERT, PLATE_FRAG, PLATE_STRIDE, PLATE_VERT } from './shaders/plate';
import { PARTICLE_FRAG, PARTICLE_STRIDE, PARTICLE_VERT } from './shaders/particles';
import { SPRITE_FRAG, SPRITE_STRIDE, SPRITE_VERT } from './shaders/sprite';
import {
  BLOOM_DOWN_FRAG,
  BLOOM_PREFILTER_FRAG,
  BLOOM_UP_FRAG,
  COMPOSITE_FRAG,
} from './shaders/post';
import { buildRamp, COLORS, hexToRgb, MAGNITUDE_ANCHORS } from '../core/palette';

export interface CameraState {
  eye: [number, number, number];
  target: [number, number, number];
  fov: number;
}

export interface PostState {
  bloomWeight: number;
  exposure: number;
  aberration: number;
  defocus: number;
  vignette: number;
  grain: number;
  groundGlow: number;
  fade: number;
}

export const DEFAULT_POST: PostState = {
  bloomWeight: 0.052,
  exposure: 1.0,
  aberration: 0.0,
  defocus: 0.0,
  vignette: 0.4,
  grain: 0.022,
  groundGlow: 0.35,
  fade: 1,
};

export interface ParticleGroupOptions {
  /** 0 = magnitude ramp, 1 = signed excitatory/inhibitory axis. */
  colorMode: number;
}

interface ParticleGroup {
  vao: WebGLVertexArrayObject;
  buffer: WebGLBuffer;
  capacity: number;
  count: number;
  colorMode: number;
  opacity: number;
  flow: number;
}

interface PlateBatch {
  stack: FieldStack;
  data: Float32Array;
  count: number;
}

/** Target pixel counts. Mobile is capped harder because bandwidth, not shading, is the wall. */
const MAX_PIXELS_DESKTOP = 2_300_000;
const MAX_PIXELS_MOBILE = 1_300_000;
const SCALE_STEPS = [1.0, 0.85, 0.72, 0.6, 0.5];

export class Renderer {
  readonly ctx: GLContext;
  private readonly gl: WebGL2RenderingContext;
  private readonly tri: FullscreenTriangle;

  private plateProgram: Program;
  private kernelProgram: Program;
  private particleProgram: Program;
  private prefilterProgram: Program;
  private downProgram: Program;
  private upProgram: Program;
  private compositeProgram: Program;

  private scene!: RenderTarget;
  private bloomChain: RenderTarget[] = [];

  private rampTexture: WebGLTexture;
  private kernelTexture: WebGLTexture;
  private kernelDims: [number, number] = [5, 5];

  private stacks = new Map<string, FieldStack>();
  /** Per-stack reciprocal of the largest negative magnitude. */
  private negScales = new Map<string, number>();
  /** Per-stack monochrome tint: [r, g, b, mix]. */
  private tints = new Map<string, [number, number, number, number]>();
  private plateBatches = new Map<string, PlateBatch>();
  private particleGroups = new Map<string, ParticleGroup>();

  private plateVao: WebGLVertexArrayObject;
  private plateBuffer: WebGLBuffer;
  private plateCapacity = 0;

  private kernelVao: WebGLVertexArrayObject;
  private kernelBuffer: WebGLBuffer;
  private kernelCapacity = 0;
  private kernelData = new Float32Array(0);
  private kernelCount = 0;

  private spriteProgram: Program;
  private spriteVao!: WebGLVertexArrayObject;
  private spriteBuffer!: WebGLBuffer;
  /** The shared unit-quad corner buffer. Every instanced pass binds this one. */
  private quadBuffer!: WebGLBuffer;
  private spriteCapacity = 0;
  private spriteData = new Float32Array(256 * SPRITE_STRIDE);
  private spriteCount = 0;

  private view: Mat4 = new Float32Array(16);
  private proj: Mat4 = new Float32Array(16);
  private viewProj: Mat4 = new Float32Array(16);

  private frame = 0;
  private renderScale = 1;
  private scaleIndex = 0;
  private frameTimes: number[] = [];
  private lastScaleChange = 0;
  private isMobile: boolean;

  cssWidth = 1;
  cssHeight = 1;

  /** Aspect of the render target. The camera frames content against this every frame. */
  get aspect(): number {
    return this.scene.width / Math.max(1, this.scene.height);
  }

  /**
   * World point to viewport fraction, using the same matrix the frame was drawn with.
   * Labels live in the DOM at full device pixel ratio rather than being rendered into
   * the (often downscaled) GL target, because type is what the eye judges sharpness by.
   */
  project(p: readonly [number, number, number]): { x: number; y: number; visible: boolean } {
    const m = this.viewProj;
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (w <= 0.001) return { x: 0, y: 0, visible: false };
    const x = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w;
    const y = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w;
    return { x: x * 0.5 + 0.5, y: 1 - (y * 0.5 + 0.5), visible: true };
  }

  constructor(canvas: HTMLCanvasElement) {
    const ctx = createContext(canvas);
    if (!ctx) throw new Error('WebGL2 is not available');
    this.ctx = ctx;
    const gl = ctx.gl;
    this.gl = gl;
    this.isMobile = matchMedia('(pointer: coarse)').matches;

    this.tri = new FullscreenTriangle(gl);

    // All programs are created up front. Link status is checked later, in one batch, so
    // we never block on the shader compiler mid-interaction.
    this.plateProgram = new Program(gl, PLATE_VERT, PLATE_FRAG, 'plate');
    this.kernelProgram = new Program(gl, KERNEL_VERT, KERNEL_FRAG, 'kernel');
    this.particleProgram = new Program(gl, PARTICLE_VERT, PARTICLE_FRAG, 'particle');
    this.spriteProgram = new Program(gl, SPRITE_VERT, SPRITE_FRAG, 'sprite');
    this.prefilterProgram = new Program(gl, VERT_FULLSCREEN, BLOOM_PREFILTER_FRAG, 'prefilter');
    this.downProgram = new Program(gl, VERT_FULLSCREEN, BLOOM_DOWN_FRAG, 'bloom-down');
    this.upProgram = new Program(gl, VERT_FULLSCREEN, BLOOM_UP_FRAG, 'bloom-up');
    this.compositeProgram = new Program(gl, VERT_FULLSCREEN, COMPOSITE_FRAG, 'composite');

    this.rampTexture = this.createRamp();
    this.kernelTexture = createFieldTexture(gl, 25, 64);

    const geo = this.createQuadGeometry();
    this.plateVao = geo.plateVao;
    this.plateBuffer = geo.plateBuffer;
    this.kernelVao = geo.kernelVao;
    this.kernelBuffer = geo.kernelBuffer;
    this.spriteVao = geo.spriteVao;
    this.spriteBuffer = geo.spriteBuffer;
    this.quadBuffer = geo.cornerBuf;

    const format = ctx.hdr ? gl.R11F_G11F_B10F : gl.RGBA8;
    this.scene = new RenderTarget(gl, 2, 2, format);
    for (let i = 0; i < 5; i++) this.bloomChain.push(new RenderTarget(gl, 2, 2, format));

    for (const p of [
      this.plateProgram,
      this.kernelProgram,
      this.particleProgram,
      this.spriteProgram,
      this.prefilterProgram,
      this.downProgram,
      this.upProgram,
      this.compositeProgram,
    ]) {
      p.verify();
    }
  }

  private createRamp(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error('could not create ramp texture');
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      256,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      buildRamp(MAGNITUDE_ANCHORS),
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private createQuadGeometry() {
    const gl = this.gl;
    const corners = new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
    const cornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    // -- plates: 16 floats per instance
    const plateVao = gl.createVertexArray()!;
    const plateBuffer = gl.createBuffer()!;
    gl.bindVertexArray(plateVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, plateBuffer);
    const S = PLATE_STRIDE * 4;
    const plateAttrs: [number, number, number][] = [
      [1, 3, 0],
      [2, 2, 12],
      [3, 1, 20],
      [4, 4, 24],
      [5, 4, 40],
    ];
    for (const [loc, size, offset] of plateAttrs) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, S, offset);
      gl.vertexAttribDivisor(loc, 1);
    }

    // -- kernels: 12 floats per instance
    const kernelVao = gl.createVertexArray()!;
    const kernelBuffer = gl.createBuffer()!;
    gl.bindVertexArray(kernelVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, kernelBuffer);
    const K = 12 * 4;
    const kernelAttrs: [number, number, number][] = [
      [1, 3, 0],
      [2, 2, 12],
      [3, 4, 20],
    ];
    for (const [loc, size, offset] of kernelAttrs) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, K, offset);
      gl.vertexAttribDivisor(loc, 1);
    }

    // -- sprites: 16 floats per instance
    const spriteVao = gl.createVertexArray()!;
    const spriteBuffer = gl.createBuffer()!;
    gl.bindVertexArray(spriteVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, spriteBuffer);
    const P = SPRITE_STRIDE * 4;
    const spriteAttrs: [number, number, number][] = [
      [1, 3, 0],
      [2, 2, 12],
      [3, 4, 20],
      [4, 4, 36],
    ];
    for (const [loc, size, offset] of spriteAttrs) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, P, offset);
      gl.vertexAttribDivisor(loc, 1);
    }

    gl.bindVertexArray(null);
    return { plateVao, plateBuffer, kernelVao, kernelBuffer, spriteVao, spriteBuffer, cornerBuf };
  }

  // -- resources ------------------------------------------------------------

  /** Declare a layer stack. Safe to call repeatedly with the same shape. */
  ensureStack(name: string, width: number, height: number, layers: number): FieldStack {
    const existing = this.stacks.get(name);
    if (existing && existing.width === width && existing.height === height && existing.layers === layers) {
      return existing;
    }
    existing?.dispose();
    // Any pending batch captured the old stack, whose texture has just been deleted.
    this.plateBatches.delete(name);
    const stack = new FieldStack(this.gl, width, height, layers);
    this.stacks.set(name, stack);
    return stack;
  }

  uploadStack(name: string, data: Float32Array) {
    this.stacks.get(name)?.upload(data);
  }

  setStackNegScale(name: string, scale: number) {
    this.negScales.set(name, scale);
  }

  /** Render a whole stack as a monochrome tint instead of the magnitude ramp. */
  setStackTint(name: string, color: readonly [number, number, number], mix: number) {
    this.tints.set(name, [color[0], color[1], color[2], mix]);
  }

  /** `data` is [kernelIndex][k*k]; up to 64 kernels. */
  uploadKernels(data: Float32Array, k: number, count: number) {
    const padded = new Float32Array(25 * 64);
    const n = k * k;
    for (let i = 0; i < Math.min(count, 64); i++) {
      for (let j = 0; j < n; j++) padded[i * 25 + j] = data[i * n + j];
    }
    uploadField(this.gl, this.kernelTexture, 25, 64, padded);
    this.kernelDims = [k, k];
  }

  createParticleGroup(name: string, capacity: number, options: ParticleGroupOptions) {
    const gl = this.gl;
    const existing = this.particleGroups.get(name);
    if (existing) {
      gl.deleteVertexArray(existing.vao);
      gl.deleteBuffer(existing.buffer);
    }
    const vao = gl.createVertexArray()!;
    const buffer = gl.createBuffer()!;

    gl.bindVertexArray(vao);
    // Reuse the shared quad rather than allocating a private one. A per-call buffer
    // could not be reached by the replacement branch above, so every run leaked one —
    // and this runs twice per Reveal, for the lifetime of the page.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, capacity * PARTICLE_STRIDE * 4, gl.DYNAMIC_DRAW);
    const P = PARTICLE_STRIDE * 4;
    const attrs: [number, number, number][] = [
      [1, 3, 0],
      [2, 3, 12],
      [3, 3, 24],
      [4, 4, 36],
      [5, 3, 52],
    ];
    for (const [loc, size, offset] of attrs) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, P, offset);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);

    this.particleGroups.set(name, {
      vao,
      buffer,
      capacity,
      count: 0,
      colorMode: options.colorMode,
      opacity: 1,
      flow: 1,
    });
  }

  setParticleData(name: string, data: Float32Array, count: number) {
    const group = this.particleGroups.get(name);
    if (!group) return;
    const gl = this.gl;
    group.count = Math.min(count, group.capacity);
    gl.bindBuffer(gl.ARRAY_BUFFER, group.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data.subarray(0, group.count * PARTICLE_STRIDE));
  }

  setParticleState(name: string, opacity: number, flow: number) {
    const group = this.particleGroups.get(name);
    if (!group) return;
    group.opacity = opacity;
    group.flow = flow;
  }

  // -- per-frame plate submission ------------------------------------------

  /** Clear all submitted geometry. Call once at the start of a frame. */
  beginFrame() {
    for (const batch of this.plateBatches.values()) batch.count = 0;
    this.kernelCount = 0;
    this.spriteCount = 0;
  }

  /** A disc, rounded bar or ring. `color` is sRGB in [0,1]. */
  sprite(
    center: readonly [number, number, number],
    size: readonly [number, number],
    color: readonly [number, number, number],
    opacity: number,
    options?: { mode?: number; radius?: number; softness?: number; intensity?: number },
  ) {
    if (opacity <= 0.003) return;
    const need = (this.spriteCount + 1) * SPRITE_STRIDE;
    if (need > this.spriteData.length) {
      const grown = new Float32Array(this.spriteData.length * 2);
      grown.set(this.spriteData);
      this.spriteData = grown;
    }
    const o = this.spriteCount * SPRITE_STRIDE;
    const d = this.spriteData;
    d[o] = center[0];
    d[o + 1] = center[1];
    d[o + 2] = center[2];
    d[o + 3] = size[0];
    d[o + 4] = size[1];
    d[o + 5] = color[0];
    d[o + 6] = color[1];
    d[o + 7] = color[2];
    d[o + 8] = opacity;
    d[o + 9] = options?.mode ?? 0;
    d[o + 10] = options?.radius ?? 0.02;
    d[o + 11] = options?.softness ?? 0.02;
    d[o + 12] = options?.intensity ?? 1.4;
    this.spriteCount++;
  }

  /**
   * Submit one feature map. Values are grouped by stack so each stack becomes one
   * instanced draw call.
   */
  plate(
    stackName: string,
    layer: number,
    center: readonly [number, number, number],
    size: readonly [number, number],
    style: { opacity: number; valueScale: number; cellBias?: number; grid?: number },
    state?: { signedMix?: number; sweep?: number; highlight?: number; dissolve?: number },
  ) {
    const stack = this.stacks.get(stackName);
    if (!stack || style.opacity <= 0.002) return;

    let batch = this.plateBatches.get(stackName);
    if (!batch) {
      batch = { stack, data: new Float32Array(64 * PLATE_STRIDE), count: 0 };
      this.plateBatches.set(stackName, batch);
    }
    if ((batch.count + 1) * PLATE_STRIDE > batch.data.length) {
      const grown = new Float32Array(batch.data.length * 2);
      grown.set(batch.data);
      batch.data = grown;
    }
    const o = batch.count * PLATE_STRIDE;
    const d = batch.data;
    d[o] = center[0];
    d[o + 1] = center[1];
    d[o + 2] = center[2];
    d[o + 3] = size[0];
    d[o + 4] = size[1];
    d[o + 5] = layer;
    d[o + 6] = style.opacity;
    d[o + 7] = style.valueScale;
    d[o + 8] = style.cellBias ?? 1;
    d[o + 9] = style.grid ?? 0;
    d[o + 10] = state?.signedMix ?? 0;
    d[o + 11] = state?.sweep ?? 1;
    d[o + 12] = state?.highlight ?? 0;
    d[o + 13] = state?.dissolve ?? 0;
    batch.count++;
  }

  kernel(
    index: number,
    center: readonly [number, number, number],
    size: number,
    opacity: number,
    scale: number,
    highlight = 0,
  ) {
    if (opacity <= 0.002) return;
    const need = (this.kernelCount + 1) * 12;
    if (need > this.kernelData.length) {
      const grown = new Float32Array(Math.max(need, 64 * 12));
      grown.set(this.kernelData);
      this.kernelData = grown;
    }
    const o = this.kernelCount * 12;
    const d = this.kernelData;
    d[o] = center[0];
    d[o + 1] = center[1];
    d[o + 2] = center[2];
    d[o + 3] = size;
    d[o + 4] = size;
    d[o + 5] = opacity;
    d[o + 6] = scale;
    d[o + 7] = index;
    d[o + 8] = highlight;
    this.kernelCount++;
  }

  // -- sizing ---------------------------------------------------------------

  resize(cssWidth: number, cssHeight: number) {
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    const cap = this.isMobile ? 1.5 : 2;
    const dpr = Math.min(window.devicePixelRatio || 1, cap);
    const budget = this.isMobile ? MAX_PIXELS_MOBILE : MAX_PIXELS_DESKTOP;
    const raw = cssWidth * cssHeight * dpr * dpr;
    const budgetScale = Math.min(1, Math.sqrt(budget / Math.max(raw, 1)));

    const w = Math.max(2, Math.round(cssWidth * dpr * budgetScale * this.renderScale));
    const h = Math.max(2, Math.round(cssHeight * dpr * budgetScale * this.renderScale));

    // Assigning canvas.width reallocates and clears the drawing buffer even when the
    // value is unchanged, so a resize storm (a mobile URL bar collapsing, a desktop
    // window drag) would otherwise pay a full realloc per event.
    if (w === this.ctx.canvas.width && h === this.ctx.canvas.height) return;

    this.ctx.canvas.width = w;
    this.ctx.canvas.height = h;
    this.scene.resize(w, h);

    // The bloom pyramid starts at quarter linear resolution and is sized so the
    // smallest mip is around a dozen pixels — that keeps the glow radius perceptually
    // constant instead of changing when the adaptive scale steps.
    let bw = Math.max(2, w >> 2);
    let bh = Math.max(2, h >> 2);
    for (const rt of this.bloomChain) {
      rt.resize(bw, bh);
      bw = Math.max(2, bw >> 1);
      bh = Math.max(2, bh >> 1);
    }
  }

  /**
   * Adaptive resolution driven by the median of recent frames. Median rather than mean
   * because a single GC pause would otherwise drive the controller to the floor, and
   * quantised steps with a cooldown because reallocating render targets is itself a hitch.
   */
  private adapt(frameMs: number) {
    this.frameTimes.push(frameMs);
    if (this.frameTimes.length < 40) return;
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    if (this.frame - this.lastScaleChange < 90) return;

    const sorted = [...this.frameTimes].sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];

    let next = this.scaleIndex;
    if (median > 15 && this.scaleIndex < SCALE_STEPS.length - 1) next = this.scaleIndex + 1;
    else if (median < 10.5 && this.scaleIndex > 0) next = this.scaleIndex - 1;

    if (next !== this.scaleIndex) {
      this.scaleIndex = next;
      this.renderScale = SCALE_STEPS[next];
      this.lastScaleChange = this.frame;
      this.frameTimes.length = 0;
      this.resize(this.cssWidth, this.cssHeight);
    }
  }

  // -- drawing --------------------------------------------------------------

  /**
   * Set the camera for this frame. Called before geometry is submitted so that anything
   * needing to project a world point (the annotation layer) uses this frame's matrix
   * rather than the previous one, which would lag visibly during a camera move.
   */
  setCamera(camera: CameraState) {
    perspective(camera.fov, this.aspect, 0.1, 200, this.proj);
    lookAt(camera.eye, camera.target, [0, 1, 0], this.view);
    multiply(this.proj, this.view, this.viewProj);
  }

  render(post: PostState, time: number, frameMs: number) {
    const gl = this.gl;
    this.frame++;
    this.adapt(frameMs);

    // -- scene pass, fully additive
    this.scene.bind(true);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    const pos = hexToRgb(COLORS.positive);
    const neg = hexToRgb(COLORS.negative);

    // plates
    if (this.plateBatches.size) {
      const p = this.plateProgram;
      p.use();
      p.m4('uViewProj', this.viewProj);
      p.v3('uPositive', pos[0], pos[1], pos[2]);
      p.v3('uNegative', neg[0], neg[1], neg[2]);
      p.f('uTime', time);
      p.tex('uRamp', 0, this.rampTexture);
      gl.bindVertexArray(this.plateVao);
      for (const [name, batch] of this.plateBatches) {
        if (batch.count === 0) continue;
        p.tex('uField', 1, batch.stack.texture, gl.TEXTURE_2D_ARRAY);
        p.v2('uFieldSize', batch.stack.width, batch.stack.height);
        p.f('uNegScale', this.negScales.get(name) ?? 1);
        const tint = this.tints.get(name);
        p.v3('uTint', tint ? tint[0] : 1, tint ? tint[1] : 1, tint ? tint[2] : 1);
        p.f('uTintMix', tint ? tint[3] : 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.plateBuffer);
        const needed = batch.count * PLATE_STRIDE;
        if (needed > this.plateCapacity) {
          gl.bufferData(gl.ARRAY_BUFFER, batch.data.byteLength, gl.DYNAMIC_DRAW);
          this.plateCapacity = batch.data.length;
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, batch.data.subarray(0, needed));
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, batch.count);
      }
    }

    // kernels
    if (this.kernelCount > 0) {
      const p = this.kernelProgram;
      p.use();
      p.m4('uViewProj', this.viewProj);
      p.v2('uKernelSize', this.kernelDims[0], this.kernelDims[1]);
      p.v3('uPositive', pos[0], pos[1], pos[2]);
      p.v3('uNegative', neg[0], neg[1], neg[2]);
      p.tex('uRamp', 0, this.rampTexture);
      // Unit 2, not 1: unit 1 holds a TEXTURE_2D_ARRAY for the plate pass, and leaving
      // two targets bound to one unit is the kind of thing that works everywhere except
      // the one driver you did not test on.
      p.tex('uKernels', 2, this.kernelTexture);
      gl.bindVertexArray(this.kernelVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.kernelBuffer);
      const needed = this.kernelCount * 12;
      if (needed > this.kernelCapacity) {
        gl.bufferData(gl.ARRAY_BUFFER, this.kernelData.byteLength, gl.DYNAMIC_DRAW);
        this.kernelCapacity = this.kernelData.length;
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.kernelData.subarray(0, needed));
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.kernelCount);
    }

    // sprites
    if (this.spriteCount > 0) {
      const p = this.spriteProgram;
      p.use();
      p.m4('uViewProj', this.viewProj);
      p.tex('uRamp', 0, this.rampTexture);
      gl.bindVertexArray(this.spriteVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.spriteBuffer);
      const needed = this.spriteCount * SPRITE_STRIDE;
      if (needed > this.spriteCapacity) {
        gl.bufferData(gl.ARRAY_BUFFER, this.spriteData.byteLength, gl.DYNAMIC_DRAW);
        this.spriteCapacity = this.spriteData.length;
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.spriteData.subarray(0, needed));
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.spriteCount);
    }

    // particles
    for (const group of this.particleGroups.values()) {
      if (group.count === 0 || group.opacity <= 0.002) continue;
      const p = this.particleProgram;
      p.use();
      p.m4('uViewProj', this.viewProj);
      p.f('uTime', time);
      p.f('uGlobalOpacity', group.opacity);
      p.f('uFlow', group.flow);
      p.v2('uProjScale', this.proj[0], this.proj[5]);
      p.i('uColorMode', group.colorMode);
      p.v3('uPositive', pos[0], pos[1], pos[2]);
      p.v3('uNegative', neg[0], neg[1], neg[2]);
      p.tex('uRamp', 0, this.rampTexture);
      gl.bindVertexArray(group.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, group.count);
    }

    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);

    // -- bloom pyramid
    const chain = this.bloomChain;
    {
      const p = this.prefilterProgram;
      p.use();
      chain[0].bind(true);
      p.tex('uSrc', 0, this.scene.texture);
      p.v2('uTexel', 1 / this.scene.width, 1 / this.scene.height);
      // Only genuinely hot pixels bloom. A low threshold made the whole frame hazy and
      // smeared the feature maps into each other, which is the opposite of what this
      // needs — the glow should mark the peaks, not fog the picture.
      p.f('uThreshold', 0.82);
      p.f('uKnee', 0.38);
      this.tri.draw();
    }
    {
      const p = this.downProgram;
      p.use();
      for (let i = 1; i < chain.length; i++) {
        chain[i].bind(true);
        p.tex('uSrc', 0, chain[i - 1].texture);
        p.v2('uHalfPixel', 0.5 / chain[i - 1].width, 0.5 / chain[i - 1].height);
        this.tri.draw();
      }
    }
    {
      // Accumulate each smaller mip into the next larger one with additive blending.
      // Sampling the destination while writing it would be a feedback loop, so the
      // blend hardware does the combining instead of the shader.
      const p = this.upProgram;
      p.use();
      p.f('uScatter', 0.6);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      for (let i = chain.length - 1; i > 0; i--) {
        chain[i - 1].bind(false);
        p.tex('uSrc', 0, chain[i].texture);
        p.v2('uHalfPixel', 0.5 / chain[i].width, 0.5 / chain[i].height);
        this.tri.draw();
      }
      gl.disable(gl.BLEND);
    }

    // -- composite, one pass to the screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.scene.width, this.scene.height);
    const c = this.compositeProgram;
    c.use();
    c.tex('uScene', 0, this.scene.texture);
    c.tex('uBloom', 1, chain[0].texture);
    c.v2('uResolution', this.scene.width, this.scene.height);
    c.f('uTime', time);
    c.f('uFrame', this.frame);
    c.f('uBloomWeight', post.bloomWeight);
    c.f('uExposure', post.exposure);
    c.f('uAberration', post.aberration);
    c.f('uDefocus', post.defocus);
    c.f('uVignette', post.vignette);
    c.f('uGrain', post.grain);
    c.f('uFade', post.fade);
    const ground = hexToRgb(COLORS.ground);
    c.v3('uGround', ground[0], ground[1], ground[2]);
    c.f('uGroundGlow', post.groundGlow);
    this.tri.draw();
    gl.bindVertexArray(null);
  }
}
