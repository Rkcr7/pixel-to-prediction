/**
 * A small WebGL2 layer. No scene graph, no material system — this is a 2.5D compositing
 * problem, and a general-purpose engine would cost more bytes than the whole app.
 *
 * The rules encoded here come from what actually costs frames on mid-range mobile:
 * never check link status right after linkProgram (it blocks on the compiler thread),
 * always clear a freshly bound framebuffer (otherwise tile-based GPUs load the previous
 * contents from main memory), and never call getError or readPixels in the frame loop.
 */

export interface GLContext {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  /** True when EXT_color_buffer_float is present, which gates the whole HDR path. */
  hdr: boolean;
  /** Max point size; Apple silicon reports 64 where most GPUs report 512+. */
  maxPointSize: number;
}

export function createContext(canvas: HTMLCanvasElement): GLContext | null {
  const gl = canvas.getContext('webgl2', {
    alpha: false, // avoids the browser compositing the whole canvas with the page
    depth: false, // additive rendering is order independent; we never need a depth buffer
    stencil: false,
    antialias: false, // MSAA resolve is pure bandwidth; the composite pass handles edges
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
    desynchronized: true,
  });
  if (!gl) return null;

  // Renderable float colour buffers. Present on ~99.7% of WebGL2 devices; without it we
  // fall back to a correct LDR path rather than maintaining a parallel pipeline.
  const hdr = !!gl.getExtension('EXT_color_buffer_float');
  const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array;

  return { gl, canvas, hdr, maxPointSize: range ? range[1] : 64 };
}

// GLSL ES 3.0 supplies a default precision for float, int and sampler2D but *not* for
// sampler2DArray, so it has to be declared or every shader touching a feature-map stack
// fails to compile.
const HEADER =
  '#version 300 es\n' +
  'precision highp float;\n' +
  'precision highp int;\n' +
  'precision highp sampler2DArray;\n';

export class Program {
  readonly program: WebGLProgram;
  private readonly uniforms = new Map<string, WebGLUniformLocation | null>();
  private checked = false;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    vertexSource: string,
    fragmentSource: string,
    readonly label = 'program',
  ) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, HEADER + vertexSource, `${label}:vert`);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, HEADER + fragmentSource, `${label}:frag`);
    const program = gl.createProgram();
    if (!program) throw new Error(`${label}: could not create program`);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    // Deliberately not calling getProgramParameter here: that forces a synchronous wait
    // on the shader compiler and is the classic cause of a long hitch at first paint.
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.program = program;
  }

  /** Call once, after every program in the app has been linked. */
  verify() {
    if (this.checked) return;
    this.checked = true;
    const gl = this.gl;
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(`${this.label}: link failed\n${gl.getProgramInfoLog(this.program)}`);
    }
  }

  use() {
    this.gl.useProgram(this.program);
  }

  loc(name: string): WebGLUniformLocation | null {
    let l = this.uniforms.get(name);
    if (l === undefined) {
      l = this.gl.getUniformLocation(this.program, name);
      this.uniforms.set(name, l);
    }
    return l;
  }

  f(name: string, v: number) {
    this.gl.uniform1f(this.loc(name), v);
  }
  i(name: string, v: number) {
    this.gl.uniform1i(this.loc(name), v);
  }
  v2(name: string, x: number, y: number) {
    this.gl.uniform2f(this.loc(name), x, y);
  }
  v3(name: string, x: number, y: number, z: number) {
    this.gl.uniform3f(this.loc(name), x, y, z);
  }
  v4(name: string, x: number, y: number, z: number, w: number) {
    this.gl.uniform4f(this.loc(name), x, y, z, w);
  }
  m4(name: string, m: Float32Array) {
    this.gl.uniformMatrix4fv(this.loc(name), false, m);
  }
  /** Bind a texture to `unit` and point the sampler uniform at it. */
  tex(name: string, unit: number, texture: WebGLTexture | null, target = 0x0de1 /* TEXTURE_2D */) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, texture);
    gl.uniform1i(this.loc(name), unit);
  }
}

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error(`${label}: could not create shader`);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? '';
    const numbered = source
      .split('\n')
      .map((l, i) => `${String(i + 1).padStart(3)} | ${l}`)
      .join('\n');
    throw new Error(`${label}: compile failed\n${log}\n${numbered}`);
  }
  return shader;
}

/**
 * A render target. Uses R11F_G11F_B10F where available: half the bytes of RGBA16F,
 * filterable and blendable in core WebGL2, and bloom has no use for alpha.
 */
export class RenderTarget {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
  width = 0;
  height = 0;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    width: number,
    height: number,
    private readonly internalFormat: number,
    private readonly linear = true,
  ) {
    const tex = gl.createTexture();
    const fb = gl.createFramebuffer();
    if (!tex || !fb) throw new Error('could not create render target');
    this.texture = tex;
    this.framebuffer = fb;
    this.resize(width, height);
  }

  resize(width: number, height: number) {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;

    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      this.internalFormat,
      w,
      h,
      0,
      this.internalFormat === gl.RGBA8 ? gl.RGBA : gl.RGB,
      this.internalFormat === gl.RGBA8 ? gl.UNSIGNED_BYTE : gl.FLOAT,
      null,
    );
    const filter = this.linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      this.texture,
      0,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Bind and clear. Clearing is not optional: skipping it makes tile-based GPUs
   *  load the whole previous framebuffer from main memory before the first draw. */
  bind(clear = true) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, this.width, this.height);
    if (clear) {
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
  }

  dispose() {
    this.gl.deleteTexture(this.texture);
    this.gl.deleteFramebuffer(this.framebuffer);
  }
}

/**
 * A stack of single-channel float planes — one per feature map. A 2D array texture keeps
 * the shader honest (`texture(s, vec3(uv, layer))`) instead of forcing atlas arithmetic
 * into every sampling site.
 */
export class FieldStack {
  texture: WebGLTexture;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    readonly width: number,
    readonly height: number,
    readonly layers: number,
  ) {
    const tex = gl.createTexture();
    if (!tex) throw new Error('could not create field stack');
    this.texture = tex;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    // R16F is filterable and blendable in core WebGL2. R32F is neither without an
    // extension that 8.5% of devices lack, and 10 bits of mantissa is ample for a
    // value we are about to map through a colour ramp.
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R16F, width, height, layers);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** `data` is channel-major: layer 0's plane, then layer 1's, and so on. */
  upload(data: Float32Array) {
    const gl = this.gl;
    const need = this.width * this.height * this.layers;
    if (data.length !== need) {
      throw new Error(`FieldStack expected ${need} values, got ${data.length}`);
    }
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texture);
    gl.texSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      0,
      0,
      0,
      0,
      this.width,
      this.height,
      this.layers,
      gl.RED,
      gl.FLOAT,
      data,
    );
  }

  dispose() {
    this.gl.deleteTexture(this.texture);
  }
}

/** A plain 2D single-channel float texture, for one-off fields like saliency. */
export function createFieldTexture(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error('could not create texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16F, width, height);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

export function uploadField(
  gl: WebGL2RenderingContext,
  tex: WebGLTexture,
  width: number,
  height: number,
  data: Float32Array,
) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.FLOAT, data);
}

/**
 * A single triangle covering the viewport. Preferred over a quad: a quad's diagonal
 * produces a screen-wide seam of duplicated 2x2 helper invocations.
 */
export class FullscreenTriangle {
  private vao: WebGLVertexArrayObject;

  constructor(private readonly gl: WebGL2RenderingContext) {
    const vao = gl.createVertexArray();
    const buf = gl.createBuffer();
    if (!vao || !buf) throw new Error('could not create fullscreen triangle');
    this.vao = vao;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  draw() {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

export const VERT_FULLSCREEN = `
layout(location = 0) in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;
