/**
 * The post chain.
 *
 * Bloom is a progressive downsample/upsample pyramid rather than bright-pass plus
 * separable gaussian: successive tent filters converge to a wide gaussian, and the
 * geometric area series makes it nearly free. The chain starts at quarter linear
 * resolution and never comes back up — bloom is low-frequency by definition, and the
 * upsample into full res costs more than the entire rest of the pyramid.
 *
 * Composite is deliberately ONE pass. On a tile-based mobile GPU each extra fullscreen
 * pass is a full framebuffer store plus load, so three one-tap passes cost triple what
 * one three-tap pass costs.
 */

import { GLSL_NOISE } from './common';

/** Threshold with a soft knee, so pixels do not pop as they cross the line. */
export const BLOOM_PREFILTER_FRAG = /* glsl */ `
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uTexel;      // 1 / source resolution
uniform float uThreshold;
uniform float uKnee;
out vec4 outColor;

vec3 T(vec2 uv) { return texture(uSrc, uv).rgb; }

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float karis(vec3 c) { return 1.0 / (1.0 + luma(c) * 0.25); }

void main() {
  vec2 uv = vUv;
  float x = uTexel.x, y = uTexel.y;

  vec3 a = T(uv + vec2(-2.0*x,  2.0*y));
  vec3 b = T(uv + vec2( 0.0,    2.0*y));
  vec3 c = T(uv + vec2( 2.0*x,  2.0*y));
  vec3 d = T(uv + vec2(-2.0*x,  0.0));
  vec3 e = T(uv);
  vec3 f = T(uv + vec2( 2.0*x,  0.0));
  vec3 g = T(uv + vec2(-2.0*x, -2.0*y));
  vec3 h = T(uv + vec2( 0.0,   -2.0*y));
  vec3 i = T(uv + vec2( 2.0*x, -2.0*y));
  vec3 j = T(uv + vec2(-x,  y));
  vec3 k = T(uv + vec2( x,  y));
  vec3 l = T(uv + vec2(-x, -y));
  vec3 m = T(uv + vec2( x, -y));

  // Karis average on the first downsample only: it kills single-pixel fireflies that
  // would otherwise strobe through the whole pyramid. Applying it deeper crushes real
  // highlights, so this is the only place it appears.
  vec3 g0 = (j + k + l + m) * 0.25;
  vec3 g1 = (a + b + d + e) * 0.25;
  vec3 g2 = (b + c + e + f) * 0.25;
  vec3 g3 = (d + e + g + h) * 0.25;
  vec3 g4 = (e + f + h + i) * 0.25;

  float w0 = karis(g0) * 0.5;
  float w1 = karis(g1) * 0.125;
  float w2 = karis(g2) * 0.125;
  float w3 = karis(g3) * 0.125;
  float w4 = karis(g4) * 0.125;
  float wsum = w0 + w1 + w2 + w3 + w4;

  vec3 col = (g0*w0 + g1*w1 + g2*w2 + g3*w3 + g4*w4) / max(wsum, 1e-5);

  // Unity-style soft knee.
  float br = max(col.r, max(col.g, col.b));
  float soft = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = (soft * soft) / (4.0 * uKnee + 1e-4);
  float mul = max(br - uThreshold, soft) / max(br, 1e-4);

  outColor = vec4(max(col * mul, vec3(0.0)), 1.0);
}
`;

/** Dual-Kawase downsample: 5 taps. */
export const BLOOM_DOWN_FRAG = /* glsl */ `
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uHalfPixel;
out vec4 outColor;

void main() {
  vec3 s = texture(uSrc, vUv).rgb * 4.0;
  s += texture(uSrc, vUv - uHalfPixel).rgb;
  s += texture(uSrc, vUv + uHalfPixel).rgb;
  s += texture(uSrc, vUv + vec2( uHalfPixel.x, -uHalfPixel.y)).rgb;
  s += texture(uSrc, vUv + vec2(-uHalfPixel.x,  uHalfPixel.y)).rgb;
  outColor = vec4(s / 8.0, 1.0);
}
`;

/**
 * Dual-Kawase upsample: 8 taps.
 *
 * The result is *additively blended* into the larger mip by the caller rather than mixed
 * in the shader. Reading the destination texture while rendering to it is a feedback loop
 * and undefined in WebGL; additive accumulation gets the same layered falloff for free and
 * is what the reference physically-based bloom implementations do.
 */
export const BLOOM_UP_FRAG = /* glsl */ `
in vec2 vUv;
uniform sampler2D uSrc;      // smaller mip
uniform vec2 uHalfPixel;
uniform float uScatter;
out vec4 outColor;

void main() {
  vec3 s  = texture(uSrc, vUv + vec2(-uHalfPixel.x * 2.0, 0.0)).rgb;
  s += texture(uSrc, vUv + vec2(-uHalfPixel.x,  uHalfPixel.y)).rgb * 2.0;
  s += texture(uSrc, vUv + vec2( 0.0, uHalfPixel.y * 2.0)).rgb;
  s += texture(uSrc, vUv + vec2( uHalfPixel.x,  uHalfPixel.y)).rgb * 2.0;
  s += texture(uSrc, vUv + vec2( uHalfPixel.x * 2.0, 0.0)).rgb;
  s += texture(uSrc, vUv + vec2( uHalfPixel.x, -uHalfPixel.y)).rgb * 2.0;
  s += texture(uSrc, vUv + vec2( 0.0, -uHalfPixel.y * 2.0)).rgb;
  s += texture(uSrc, vUv + vec2(-uHalfPixel.x, -uHalfPixel.y)).rgb * 2.0;
  s /= 12.0;

  outColor = vec4(s * uScatter, 1.0);
}
`;

export const COMPOSITE_FRAG = /* glsl */ `
in vec2 vUv;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform vec2 uResolution;
uniform float uTime;
uniform float uFrame;

uniform float uBloomWeight;
uniform float uExposure;
uniform float uAberration;   // driven by how undecided the network still is
uniform float uDefocus;      // soft rack-focus: leans on the blurred pyramid
uniform float uVignette;
uniform float uGrain;
uniform vec3  uGround;
uniform float uGroundGlow;
uniform float uFade;

out vec4 outColor;
${GLSL_NOISE}

/** Narkowicz ACES fit. Slightly oversaturates hot colour, which reads as heat. */
vec3 aces(vec3 x) {
  x *= 0.6;
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec2 uv = vUv;
  vec2 dir = uv - 0.5;

  // Radial chromatic aberration, zero at the centre. Tied to classifier uncertainty:
  // while the network is undecided the image is literally unresolved, and it snaps into
  // focus as the answer commits.
  float amt = uAberration * dot(dir, dir);
  vec3 scene;
  scene.r = texture(uScene, uv - dir * amt).r;
  scene.g = texture(uScene, uv).g;
  scene.b = texture(uScene, uv + dir * amt).b;

  vec3 bloom = texture(uBloom, uv).rgb;
  vec3 hdr = scene + bloom * uBloomWeight;
  hdr = mix(hdr, bloom * 2.2, uDefocus);
  hdr *= uExposure;

  vec3 col = aces(hdr);

  col = pow(max(col, 0.0), vec3(1.0 / 2.2));

  // Ground: a cool near-black with a faint centre lift, so bloom has something to sit
  // against and the frame never looks like a flat black rectangle.
  //
  // Added after the transfer curve, not before it. uGround comes from a hex code, which
  // is display-referred by definition, so adding it to linear light and then encoding
  // expanded it enormously: #05060A is 0.0196, and pow(0.0196 * 2.02, 1/2.2) lands at
  // 0.234 — the authored near-black reached the screen as roughly #3C4152, a flat slate
  // fog sitting behind every frame in the piece. Here the hex means what it says.
  float glow = uGroundGlow * (1.0 - smoothstep(0.0, 0.85, length(dir * vec2(uResolution.x / uResolution.y, 1.0))));
  col += uGround * (1.0 + glow * 3.0);

  float v = smoothstep(1.05, 0.32, length(dir * vec2(1.15, 1.0)));
  col *= mix(1.0, v, uVignette);

  // Film grain, weighted toward the shadows the way real film behaves.
  float g = ignAnim(gl_FragCoord.xy, uFrame) - 0.5;
  float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col += g * uGrain * mix(1.0, 0.3, lum);

  // Triangular-PDF dither at exactly one quantisation step. Without this, the dark
  // gradients in this palette band visibly on 8-bit displays.
  float r1 = ign(gl_FragCoord.xy);
  float r2 = ign(gl_FragCoord.xy + 17.31);
  col += (r1 - r2) / 255.0;

  outColor = vec4(col * uFade, 1.0);
}
`;
