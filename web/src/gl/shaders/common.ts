/**
 * Shared GLSL. Every technique here is chosen for how it behaves at 28x28 magnified 25x,
 * which is an unusual regime: naive bilinear shows a diamond lattice in every derivative,
 * and naive nearest shows staircase edges.
 */

export const GLSL_COLOR = /* glsl */ `
uniform sampler2D uRamp;

vec3 srgbToLinear(vec3 c) {
  // Cheap and accurate enough; the exact piecewise curve is not worth the branch here.
  return c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878);
}

/**
 * Map a normalised magnitude to linear HDR colour.
 *
 * Two curves, both tuned for the value that actually occurs rather than for the extremes.
 * A feature map's typical activation sits around half the layer maximum, and a linear
 * lookup puts that in the ramp's dark purple with a cubic intensity term that leaves it
 * below the bloom threshold — so the common case renders muddy while only the rare peak
 * looks good. The gamma lifts mid magnitudes into the warm part of the ramp, and the
 * quadratic intensity keeps real headroom at the top without crushing the middle.
 */
vec3 magnitudeColor(float t) {
  t = clamp(t, 0.0, 1.0);
  float shaped = pow(t, 0.68);
  vec3 srgb = texture(uRamp, vec2(shaped, 0.5)).rgb;
  float hdr = 0.42 + 2.35 * t * t;
  return srgbToLinear(srgb) * hdr;
}

/** Signed value in [-1,1] to the excitatory/inhibitory axis, at matched lightness. */
vec3 signColor(float v, vec3 pos, vec3 neg) {
  float m = abs(clamp(v, -1.0, 1.0));
  vec3 base = v >= 0.0 ? pos : neg;
  return srgbToLinear(base) * (0.25 + 3.0 * m * m);
}
`;

export const GLSL_FIELD = /* glsl */ `
/**
 * Bicubic B-spline in four hardware bilinear taps.
 *
 * B-spline rather than Catmull-Rom on purpose: Catmull-Rom is sharper but has negative
 * outer lobes, which ring. Ringing around a bright cell would invent activation that the
 * network never produced, and everything on screen here is supposed to be true.
 */
float bicubicLayer(sampler2DArray tex, vec2 uv, float layer, vec2 texSize) {
  vec2 invSize = 1.0 / texSize;
  vec2 p = uv * texSize;
  vec2 tc = floor(p - 0.5) + 0.5;
  vec2 f = p - tc;
  vec2 f2 = f * f;
  vec2 f3 = f2 * f;

  // Cubic B-spline weights. They must be non-negative for the four-tap collapse below
  // to be valid: each tap position is tc + w1/(w0+w1), which is only a lerp inside the
  // intended texel pair while both weights are positive. Catmull-Rom weights look
  // similar and are sharper, but its outer lobes are negative, which pushes the taps
  // outside their pair and silently degrades the whole filter to plain bilinear.
  const float S = 1.0 / 6.0;
  vec2 w0 = S * (-f3 + 3.0 * f2 - 3.0 * f + 1.0);
  vec2 w1 = S * (3.0 * f3 - 6.0 * f2 + 4.0);
  vec2 w3 = S * f3;
  vec2 w2 = 1.0 - w0 - w1 - w3;

  vec2 s0 = w0 + w1;
  vec2 s1 = w2 + w3;
  vec2 t0 = (tc - 1.0 + w1 / s0) * invSize;
  vec2 t1 = (tc + 1.0 + w3 / s1) * invSize;

  return (texture(tex, vec3(t0.x, t0.y, layer)).r * s0.x +
          texture(tex, vec3(t1.x, t0.y, layer)).r * s1.x) * s0.y +
         (texture(tex, vec3(t0.x, t1.y, layer)).r * s0.x +
          texture(tex, vec3(t1.x, t1.y, layer)).r * s1.x) * s1.y;
}

/**
 * Antialiased nearest neighbour. Snaps to texel centres but keeps exactly one screen
 * pixel of blend at each cell boundary, so discrete cells read crisply without jaggies —
 * and it degrades to plain bilinear on its own as the plate shrinks, because fwidth grows.
 */
vec2 pixelAAUv(vec2 uv, vec2 texSize) {
  vec2 p = uv * texSize;
  vec2 i = floor(p) + 0.5;
  vec2 f = p - i;
  vec2 w = fwidth(p);
  vec2 aa = clamp(f / max(w, vec2(1e-5)), -0.5, 0.5);
  return (i + aa) / texSize;
}

/**
 * The field as drawn. cellBias lets a stage force the discrete look (to say "these are
 * numbers in a grid") or the continuous look (to say "this is a signal"); the fwidth term
 * handles everything in between automatically.
 */
float sampleField(sampler2DArray tex, vec2 uv, float layer, vec2 texSize, float cellBias) {
  float smoothV = bicubicLayer(tex, uv, layer, texSize);
  float cellV = texture(tex, vec3(pixelAAUv(uv, texSize), layer)).r;
  float texelsPerPixel = length(fwidth(uv * texSize));
  float cellness = clamp(smoothstep(0.40, 0.10, texelsPerPixel) * cellBias, 0.0, 1.0);
  return mix(smoothV, cellV, cellness);
}

/** Screen-space cell grid, constant width at any zoom. Returns coverage in [0,1]. */
float cellGrid(vec2 uv, vec2 texSize, float width) {
  vec2 p = uv * texSize;
  vec2 g = abs(fract(p) - 0.5);
  vec2 w = max(fwidth(p), vec2(1e-5));
  float d = min(g.x / w.x, g.y / w.y);
  return 1.0 - smoothstep(0.0, width, d);
}
`;

export const GLSL_NOISE = /* glsl */ `
/** Interleaved gradient noise. Three ALU ops, no texture fetch, low discrepancy. */
float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

/** Animated without destroying the low-discrepancy property. */
float ignAnim(vec2 p, float frame) {
  return ign(p + 5.588238 * mod(frame, 64.0));
}

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  return fract(p * (p + p));
}

vec2 hash21(float p) {
  vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
`;
