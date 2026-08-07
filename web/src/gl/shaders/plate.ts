/**
 * Feature-map plates: one instanced quad per channel, sampling its own layer of a
 * texture array.
 *
 * The ReLU treatment is the important part of the fragment shader. Positive values are
 * drawn *identically* whether or not ReLU has been applied — because ReLU does not
 * change them. Only the negative half is affected, and it is extinguished to nothing
 * rather than dimmed. Making survivors brighter would teach the wrong thing.
 */

import { GLSL_COLOR, GLSL_FIELD, GLSL_NOISE } from './common';

/** Floats per instance. */
export const PLATE_STRIDE = 16;

export const PLATE_VERT = /* glsl */ `
layout(location = 0) in vec2 aCorner;      // -0.5..0.5 quad
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec2 aSize;
layout(location = 3) in float aLayer;
layout(location = 4) in vec4 aStyle;       // opacity, valueScale, cellBias, grid
layout(location = 5) in vec4 aState;       // signedMix, sweep, highlight, dissolve

uniform mat4 uViewProj;

out vec2 vUv;
out float vLayer;
flat out vec4 vStyle;
flat out vec4 vState;
out vec2 vLocal;

void main() {
  // Texture row 0 is the TOP of the image, but +Y is up in world space. Flipping V here
  // keeps every field the right way up without each call site having to remember.
  vUv = vec2(aCorner.x + 0.5, 0.5 - aCorner.y);
  vLocal = aCorner;
  vLayer = aLayer;
  vStyle = aStyle;
  vState = aState;
  vec3 world = aCenter + vec3(aCorner * aSize, 0.0);
  gl_Position = uViewProj * vec4(world, 1.0);
}
`;

export const PLATE_FRAG = /* glsl */ `
in vec2 vUv;
in float vLayer;
flat in vec4 vStyle;
flat in vec4 vState;
in vec2 vLocal;

uniform sampler2DArray uField;
uniform vec2 uFieldSize;
uniform vec3 uPositive;
uniform vec3 uNegative;
uniform float uTime;
/** Reciprocal of the layer's largest NEGATIVE magnitude, kept separate from the positive
 *  scale because a layer's extremes are rarely symmetric: conv2 here runs -8.5 to +5.1,
 *  and normalising both halves by 8.5 leaves every real activation dim. */
uniform float uNegScale;
/** Optional monochrome tint for a whole stack, used to render "your ink" as a cool
 *  reference layer so a warm attribution overlay reads as a separate thing on top of it
 *  rather than merging into one glowing shape. */
uniform vec3 uTint;
uniform float uTintMix;

out vec4 outColor;

${GLSL_COLOR}
${GLSL_FIELD}
${GLSL_NOISE}

void main() {
  float opacity   = vStyle.x;
  float valueScale= vStyle.y;
  float cellBias  = vStyle.z;
  float gridAmount= vStyle.w;
  float signedMix = vState.x;
  float sweep     = vState.y;
  float highlight = vState.z;
  float dissolve  = vState.w;

  if (opacity <= 0.001) discard;

  float raw = sampleField(uField, vUv, vLayer, uFieldSize, cellBias);
  float n = raw;

  // The sweep is the convolution reading down the plate. Everything ahead of the head
  // is not yet computed, so it is simply absent rather than dimmed. vUv.y is 0 at the
  // top, so the head travels 0 -> 1 downward, the way a person reads.
  float head = sweep * 1.2 - 0.1;
  float reveal = 1.0 - smoothstep(head, head + 0.09, vUv.y);
  reveal = mix(1.0, reveal, step(sweep, 0.999));

  vec3 col;
  if (n >= 0.0) {
    // Positive response, normalised by the layer's largest positive value so the
    // strongest activation actually reaches the top of the ramp.
    float t = clamp(n * valueScale, 0.0, 1.0);
    col = magnitudeColor(t);
    if (uTintMix > 0.001) {
      // Must be exactly zero at t = 0. Under additive blending any non-zero floor turns
      // the empty part of the plate into a visible grey rectangle.
      col = mix(col, srgbToLinear(uTint) * (0.55 * t + 1.7 * t * t), uTintMix);
    }
  } else {
    // Negative signal: visible only while we are showing the pre-activation field,
    // and extinguished, not dimmed, as ReLU is applied.
    float cut = 1.0 - dissolve;
    // A little noise in the cut so the extinction reads as many small lights going out
    // rather than one uniform fade.
    float grain = ign(gl_FragCoord.xy * 0.7 + vLayer * 31.0);
    cut = smoothstep(grain * 0.55, grain * 0.55 + 0.45, cut);

    // A conv layer's background sits at its bias, which is usually mildly negative, so a
    // linear mapping paints every plate as one flat coral card and buries the structure.
    // Shaping the low end down keeps that background as a quiet wash while genuinely
    // strong negative responses still read.
    float m = clamp(-n * uNegScale, 0.0, 1.0);
    float shaped = pow(m, 1.9);
    col = srgbToLinear(uNegative) * shaped * 1.5 * signedMix * cut;
  }

  // A hair of light right at the sweep head, so the eye tracks the reading position.
  float edge = exp(-pow((vUv.y - head) * 15.0, 2.0)) * step(0.001, sweep) * step(sweep, 0.999);
  col += vec3(0.42, 0.62, 1.0) * edge * 1.35;

  if (gridAmount > 0.001) {
    float g = cellGrid(vUv, uFieldSize, 1.1);
    col += vec3(0.10, 0.13, 0.22) * g * gridAmount;
  }

  col *= 1.0 + highlight * 1.4;
  outColor = vec4(col * opacity * reveal, 1.0);
}
`;

/**
 * The learned kernels, drawn as small signed tiles. These are the actual weights the
 * trainer produced, not an illustration of a filter.
 */
export const KERNEL_VERT = /* glsl */ `
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec2 aSize;
layout(location = 3) in vec4 aStyle;  // opacity, scale, index, highlight

uniform mat4 uViewProj;
out vec2 vUv;
flat out vec4 vStyle;

void main() {
  vUv = vec2(aCorner.x + 0.5, 0.5 - aCorner.y);
  vStyle = aStyle;
  gl_Position = uViewProj * vec4(aCenter + vec3(aCorner * aSize, 0.0), 1.0);
}
`;

export const KERNEL_FRAG = /* glsl */ `
in vec2 vUv;
flat in vec4 vStyle;

uniform sampler2D uKernels;   // width = k*k, height = number of kernels
uniform vec2 uKernelSize;     // (k, k)
uniform vec3 uPositive;
uniform vec3 uNegative;

out vec4 outColor;
${GLSL_COLOR}

void main() {
  float opacity = vStyle.x;
  float scale = vStyle.y;
  float index = vStyle.z;
  float highlight = vStyle.w;
  if (opacity <= 0.001) discard;

  vec2 cell = floor(vUv * uKernelSize);
  cell = clamp(cell, vec2(0.0), uKernelSize - 1.0);
  float flat_i = cell.y * uKernelSize.x + cell.x;
  // The kernel texture is always 25 wide, whatever k is: uploadKernels pads every row
  // to a fixed stride. Normalising by k*k instead happens to be right for 5x5 and
  // silently samples the wrong weights for any other kernel size.
  const float texW = 25.0;
  float v = texture(uKernels, vec2((flat_i + 0.5) / texW, (index + 0.5) / 64.0)).r * scale;

  // Rounded cell so a 5x5 kernel reads as a set of taps, not a texture.
  vec2 f = fract(vUv * uKernelSize) - 0.5;
  float d = max(abs(f.x), abs(f.y));
  float cellMask = 1.0 - smoothstep(0.34, 0.46, d);

  vec3 col = signColor(v, uPositive, uNegative) * cellMask;
  col *= 1.0 + highlight * 1.2;
  outColor = vec4(col * opacity, 1.0);
}
`;
