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
  // Pad the quad by the distance the halo actually travels, not by a fraction of the
  // sprite.
  //
  // The fragment shader's halo decays over min(halfW, halfH) * 0.55, which is an
  // absolute world distance and is the same on both axes. Scaling the quad by a constant
  // 1.35 gave each axis a margin proportional to its own half-extent instead, so a long
  // thin bar got 0.09 units of room across its width for a glow that needs about 0.43:
  // the falloff was still at half strength where the geometry stopped and was cut dead
  // there, which is why every bar wore a hard-edged rectangular box. Tall sprites hid it,
  // because their vertical margin happened to be generous enough.
  //
  // Just enough margin for the anti-aliased edge.
  //
  // The sprite paints nothing outside its own shape any more, so the quad does not need
  // to reserve room for a halo. Whatever bleed a bright sprite has comes from the bloom
  // in the post pass, which is a full-screen effect: it reads the HDR buffer, so it needs
  // no geometry at all and cannot be clipped into a rectangle by one.
  vec2 halfSize = aSize * 0.5;
  vec2 grown = halfSize + vec2(max(aShape.z, 0.0001) * 2.0 + 0.006);

  vLocal = aCorner * 2.0 * grown;
  vHalf = halfSize;
  vColor = aColor;
  vShape = aShape;
  gl_Position = uViewProj * vec4(aCenter + vec3(aCorner * 2.0 * grown, 0.0), 1.0);
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
    // Filled, and nothing outside the shape.
    //
    // There used to be an exponential halo painted here. It was a mistake twice over: it
    // was clipped into a hard-edged rectangle by the sprite's own quad, and once that was
    // fixed it buried every bar in a blob of its own light. Both problems come from the
    // same place, which is that a glow is not a property of one object. It is what a
    // bright thing does to the image around it, so it belongs to the post pass, where the
    // bloom pyramid already reads the HDR buffer and spreads brightness with no geometry
    // to be clipped by. A bar brighter than 1.0 still glows; the glow is just no longer
    // drawn by the bar itself.
    alpha = 1.0 - smoothstep(-softness, softness, d);
  }

  if (alpha < 0.004) discard;
  vec3 col = srgbToLinear(vColor.rgb) * intensity;
  outColor = vec4(col * alpha * opacity, 1.0);
}
`;
