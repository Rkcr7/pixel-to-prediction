/**
 * Sprites: discs, rounded bars and rings, drawn from signed distance fields so they stay
 * clean at any size and cost one instanced draw call for the whole set.
 *
 * These carry the parts of the story that are not fields — hidden units, logit bars, the
 * fixed budget of light that softmax divides, and the frame around the crop region.
 */

import { GLSL_COLOR } from './common';

export const SPRITE_STRIDE = 16;

export const SPRITE_DISC = 0;
export const SPRITE_BAR = 1;
export const SPRITE_RING = 2;

export const SPRITE_VERT = /* glsl */ `
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec2 aSize;
layout(location = 3) in vec4 aColor;    // rgb in sRGB, a = opacity
layout(location = 4) in vec4 aShape;    // mode, radius, softness, intensity

uniform mat4 uViewProj;

out vec2 vLocal;
out vec2 vHalf;
flat out vec4 vColor;
flat out vec4 vShape;

void main() {
  // Pad the quad so the soft edge is not clipped by the geometry.
  vec2 pad = vec2(1.0 + 0.35);
  vLocal = aCorner * aSize * pad;
  vHalf = aSize * 0.5;
  vColor = aColor;
  vShape = aShape;
  gl_Position = uViewProj * vec4(aCenter + vec3(aCorner * aSize * pad, 0.0), 1.0);
}
`;

export const SPRITE_FRAG = /* glsl */ `
in vec2 vLocal;
in vec2 vHalf;
flat in vec4 vColor;
flat in vec4 vShape;

out vec4 outColor;
${GLSL_COLOR}

float sdRoundedBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
  float mode = vShape.x;
  float radius = vShape.y;
  float softness = max(vShape.z, 0.0001);
  float intensity = vShape.w;
  float opacity = vColor.a;
  if (opacity <= 0.003) discard;

  float d;
  if (mode < 0.5) {
    d = length(vLocal / max(vHalf, vec2(1e-4))) - 1.0;
    d *= min(vHalf.x, vHalf.y);
  } else {
    d = sdRoundedBox(vLocal, vHalf, min(radius, min(vHalf.x, vHalf.y)));
  }

  float alpha;
  if (mode > 1.5) {
    // Ring: a band centred on the boundary.
    alpha = 1.0 - smoothstep(0.0, softness, abs(d) - softness * 0.5);
  } else {
    // Filled, with a soft halo outside so it reads as light rather than plastic.
    // The halo falls off relative to the sprite's own radius, not to an absolute
    // constant: scaling it by softness alone made small sprites glow uniformly across
    // their whole quad, so discs came out looking like squares.
    float radius = max(min(vHalf.x, vHalf.y), 1e-4);
    float core = 1.0 - smoothstep(-softness, softness, d);
    float halo = exp(-max(d, 0.0) / (radius * 0.55)) * 0.5;
    alpha = clamp(core + halo, 0.0, 1.6);
  }

  if (alpha < 0.004) discard;
  vec3 col = srgbToLinear(vColor.rgb) * intensity;
  outColor = vec4(col * alpha * opacity, 1.0);
}
`;
