/**
 * Particles carrying signal along real connections.
 *
 * Entirely stateless: a particle's position is `f(instanceData, time)` evaluated in the
 * vertex shader. No simulation pass, no ping-pong textures, no per-frame buffer uploads,
 * and bit-identical on every GPU — which also means a scrubbed timeline shows exactly the
 * particle positions it showed on the way forward.
 *
 * Instanced quads rather than points: gl_PointSize caps at 64 on Apple silicon where most
 * GPUs allow 512+, and point sprites clip when their centre leaves the viewport.
 */

import { GLSL_COLOR, GLSL_NOISE } from './common';

/** Floats per instance. */
export const PARTICLE_STRIDE = 16;

export const MODE_FLOW = 0;
export const MODE_AMBIENT = 1;

export const PARTICLE_VERT = /* glsl */ `
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aFrom;
layout(location = 2) in vec3 aTo;
layout(location = 3) in vec3 aCtrl;
layout(location = 4) in vec4 aA;   // seed, speed, size, value
layout(location = 5) in vec3 aB;   // opacity, mode, trail

uniform mat4 uViewProj;
uniform float uTime;
uniform float uGlobalOpacity;
uniform float uFlow;               // 0 = frozen at source, 1 = full travel
/** proj[0][0] and proj[1][1]: turns a world-space size into an NDC offset. */
uniform vec2 uProjScale;

out vec2 vLocal;
flat out float vValue;
out float vAlpha;

${GLSL_NOISE}

vec3 bezier(vec3 a, vec3 c, vec3 b, float t) {
  float mt = 1.0 - t;
  return mt * mt * a + 2.0 * mt * t * c + t * t * b;
}

void main() {
  float seed = aA.x;
  float speed = aA.y;
  float size = aA.z;
  float mode = aB.y;

  vec3 pos;
  float life = 1.0;

  if (mode < 0.5) {
    // Travelling along an edge. Phase is offset per particle so a stream reads as
    // continuous flow rather than a pulse.
    float u = fract(seed + uTime * speed);
    u *= uFlow;
    pos = bezier(aFrom, aCtrl, aTo, u);
    // Fade in on departure and out on arrival so nothing pops into or out of existence.
    life = smoothstep(0.0, 0.12, u) * (1.0 - smoothstep(0.86, 1.0, u));
    size *= 0.65 + 0.55 * life;
  } else {
    // Ambient drift. Slow, low-amplitude, and never brighter than the signal: this is
    // the medium the signal moves through, not a second thing to look at.
    vec2 h = hash21(seed * 977.0);
    float t = uTime * (0.05 + 0.05 * h.x) + seed * 6.283;
    pos = aFrom + vec3(
      sin(t * 0.9 + h.y * 6.283) * aTo.x,
      cos(t * 0.7 + h.x * 6.283) * aTo.y,
      sin(t * 0.5) * aTo.z
    );
    life = 0.6 + 0.4 * sin(t * 1.7 + seed * 20.0);
  }

  vLocal = aCorner * 2.0;
  vValue = aA.w;
  vAlpha = aB.x * life * uGlobalOpacity;

  // Billboard sized in world units.
  //
  // The offset is added to clip.xy and NOT pre-multiplied by w: the perspective divide
  // that follows turns it into an NDC offset of size*projScale/w, which is what makes a
  // distant particle small. Multiplying by w first would cancel that divide and give
  // every particle the same screen size regardless of depth.
  vec4 clip = uViewProj * vec4(pos, 1.0);
  gl_Position = clip + vec4(aCorner * size * uProjScale, 0.0, 0.0);
}
`;

export const PARTICLE_FRAG = /* glsl */ `
in vec2 vLocal;
flat in float vValue;
in float vAlpha;

uniform vec3 uPositive;
uniform vec3 uNegative;
uniform int uColorMode;   // 0 = magnitude ramp, 1 = signed axis

out vec4 outColor;
${GLSL_COLOR}

void main() {
  float r = length(vLocal);
  if (r > 1.0) discard;

  // A tight cubic falloff looks brighter than a linear falloff in a sprite twice the
  // size, and costs a quarter of the fill.
  float d = 1.0 - r;
  float glow = d * d * d;
  float core = smoothstep(0.72, 1.0, d);

  float a = vAlpha * (glow * 0.8 + core * 1.4);
  if (a < 0.004) discard;

  vec3 col = uColorMode == 1
    ? signColor(vValue, uPositive, uNegative)
    : magnitudeColor(abs(vValue));

  outColor = vec4(col * a, 1.0);
}
`;
