/**
 * Where everything sits in the world.
 *
 * The whole run happens in one continuous space with stations arranged along -Z. The
 * camera dollies down that axis exactly once, moving only *between* operations and
 * holding locked *during* them — moving the camera while a transformation is being
 * explained is the fastest way to make an animation unreadable.
 */

export type Vec3 = [number, number, number];

/** Station depths. */
export const Z = {
  input: 0,
  conv1: -8,
  conv2: -16.5,
  dense: -25,
  answer: -33,
} as const;

export const FOV = (35 * Math.PI) / 180;

/**
 * How much world space each station needs to show, and how much breathing room to leave
 * around it.
 *
 * Camera distance is derived from these against the live viewport aspect rather than
 * hard-coded, because a fixed distance that frames a 4x4 grid nicely on a 16:9 desktop
 * clips it badly on a wide monitor and catastrophically on a portrait phone.
 */
export interface StationFrame {
  z: number;
  width: number;
  height: number;
  margin: number;
  /** Vertical centre of the content, so off-centre layouts stay framed. */
  centerY: number;
}

/**
 * Station frames derived from the grid shapes actually in use, so a portrait reflow
 * automatically reframes rather than needing a second set of hand-tuned numbers.
 */
export function stationFrames(aspect: number): StationFrame[] {
  const portrait = isPortrait(aspect);
  const c1 = conv1Grid(aspect);
  const c2 = conv2Grid(aspect);
  const kernelBand = portrait ? 2.5 : 1.9;

  return [
    { z: Z.input, width: 4.6, height: 4.6, margin: 1.4, centerY: 0 },
    {
      z: Z.conv1,
      width: c1.cols * c1.cell + 0.45,
      height: c1.rows * c1.cell + kernelBand,
      margin: 1.1,
      centerY: (kernelBand - 0.5) / 2,
    },
    {
      z: Z.conv2,
      width: c2.cols * c2.cell + 0.4,
      height: c2.rows * c2.cell + 0.4,
      margin: 1.14,
      centerY: 0,
    },
    { z: Z.dense, width: 12.2, height: 7.4, margin: 1.12, centerY: 0 },
    { z: Z.answer, width: 4.8, height: 4.8, margin: 1.35, centerY: 0 },
  ];
}

/** Distance at which a box of `width` x `height` fits the frame, with margin. */
export function fitDistance(frame: StationFrame, aspect: number): number {
  const t = Math.tan(FOV / 2);
  const byHeight = (frame.height * frame.margin) / 2 / t;
  const byWidth = (frame.width * frame.margin) / 2 / (t * Math.max(aspect, 0.2));
  return Math.max(byHeight, byWidth, 3.5);
}

export interface GridSpec {
  cols: number;
  rows: number;
  cell: number;
  plate: number;
}

/**
 * Feature-map grids come in two shapes.
 *
 * Retreating the camera can only fix clipping. Wide content in a tall viewport still
 * ends up a thin band stranded in an empty frame: at iPhone portrait the 4x2 conv1 grid
 * forces the camera back to 44 world units and leaves 20 units of vertical slack. The
 * content has to change shape, so a portrait viewport gets a tall arrangement instead.
 */
export const PORTRAIT_ASPECT = 0.95;

export const isPortrait = (aspect: number) => aspect < PORTRAIT_ASPECT;

const CONV1_LANDSCAPE: GridSpec = { cols: 4, rows: 2, cell: 2.78, plate: 2.34 };
const CONV1_PORTRAIT: GridSpec = { cols: 2, rows: 4, cell: 2.78, plate: 2.34 };
const CONV2_LANDSCAPE: GridSpec = { cols: 4, rows: 4, cell: 2.02, plate: 1.82 };
const CONV2_PORTRAIT: GridSpec = { cols: 2, rows: 8, cell: 2.02, plate: 1.82 };

export const conv1Grid = (aspect: number): GridSpec =>
  isPortrait(aspect) ? CONV1_PORTRAIT : CONV1_LANDSCAPE;

export const conv2Grid = (aspect: number): GridSpec =>
  isPortrait(aspect) ? CONV2_PORTRAIT : CONV2_LANDSCAPE;

/** Landscape shapes, kept for the places that only need nominal sizes. */
export const CONV1_GRID = CONV1_LANDSCAPE;
export const CONV2_GRID = CONV2_LANDSCAPE;
export const POOL2_GRID: GridSpec = { cols: 4, rows: 4, cell: 1.06, plate: 0.98 };

/** Centre of cell `i` in a grid, laid out row-major and centred on the origin. */
export function gridCell(spec: GridSpec, i: number, z: number): Vec3 {
  const col = i % spec.cols;
  const row = Math.floor(i / spec.cols);
  const x = (col - (spec.cols - 1) / 2) * spec.cell;
  const y = -(row - (spec.rows - 1) / 2) * spec.cell;
  return [x, y, z];
}

/**
 * The eight learned kernels, in a row directly above the plates they produce, close
 * enough that the pairing reads as cause and effect rather than as two separate rows.
 */
/**
 * The learned filters, above the plates they produce.
 *
 * One row of eight in landscape. In portrait the grid is only two plates wide, so eight
 * across would be either off-frame or too small to read the taps; two rows of four keeps
 * them the same size and inside the frame.
 */
export function kernelSlot(i: number, count: number, z: number, aspect: number): Vec3 {
  const grid = conv1Grid(aspect);
  const span = grid.cols * grid.cell;
  const top = (grid.rows * grid.cell) / 2;

  if (isPortrait(aspect)) {
    const cols = Math.ceil(count / 2);
    const col = i % cols;
    const row = Math.floor(i / cols);
    return [(col - (cols - 1) / 2) * (span / cols), top + 1.7 - row * 1.0, z + 0.35];
  }
  return [(i - (count - 1) / 2) * (span / count), top + 0.98, z + 0.35];
}

// -- Stage 5: hidden units and class candidates ------------------------------

export const HIDDEN_COLS = 4;
export const HIDDEN_ROWS = 8;
export const HIDDEN_X = -0.35;

export function hiddenSlot(i: number, z: number): Vec3 {
  const col = i % HIDDEN_COLS;
  const row = Math.floor(i / HIDDEN_COLS);
  return [
    HIDDEN_X + (col - (HIDDEN_COLS - 1) / 2) * 0.46,
    -(row - (HIDDEN_ROWS - 1) / 2) * 0.58,
    z,
  ];
}

export const POOL2_BLOCK_X = -4.3;

export function pool2Slot(i: number, z: number): Vec3 {
  const c = gridCell(POOL2_GRID, i, z);
  return [c[0] + POOL2_BLOCK_X, c[1], c[2]];
}

export const CANDIDATE_X = 4.05;

/** The ten candidate digits, stacked vertically while the network is deciding. */
export function candidateSlot(digit: number, z: number): Vec3 {
  return [CANDIDATE_X, -(digit - 4.5) * 0.685, z];
}

// -- Stage 6: the bars -------------------------------------------------------

export const BAR_SPACING = 1.02;
export const BAR_WIDTH = 0.52;
export const BAR_MAX_HEIGHT = 3.1;
/** Y of the zero line. Logits can be negative, and that has to be visible. */
export const BAR_BASE_Y = -1.55;

export function barX(digit: number): number {
  return (digit - 4.5) * BAR_SPACING;
}

/** Linear interpolation between two positions. */
export function mixVec(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/**
 * Control point for the arc a particle follows between two nodes. Bowing the path
 * outward keeps parallel streams visually separable; straight lines between a dense
 * source and a dense target collapse into a hairball.
 */
export function arcControl(from: Vec3, to: Vec3, bow: number, seed: number): Vec3 {
  const mx = (from[0] + to[0]) / 2;
  const my = (from[1] + to[1]) / 2;
  const mz = (from[2] + to[2]) / 2;
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy) || 1;
  // Perpendicular in the XY plane, sign alternating by seed so streams fan both ways.
  const side = seed % 2 === 0 ? 1 : -1;
  return [mx + (-dy / len) * bow * side, my + (dx / len) * bow * side, mz + bow * 0.35];
}
