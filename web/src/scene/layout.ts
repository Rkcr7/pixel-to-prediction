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
    // The dense station turns through ninety degrees in portrait, and the frame has to
    // cover both of its beats: the stacked dot-product panels (3.1 wide, 12.7 tall
    // including headings) and the pooled-features / lattice / candidates column.
    // Width is what actually solves this one in portrait, not height: at 0.48 aspect a
    // 5.2-wide frame gives only twelve units of visible height for a panel stack that
    // spans nearly thirteen, so the top heading was cut off above the frame. The width
    // here is set by stage 6's rows, which reach from the prototype column at -2.8 to a
    // full-length bar's reading at +3.4, and it buys the height the stack needs.
    portrait
      ? { z: Z.dense, width: 7.0, height: 10.9, margin: 1.04, centerY: -0.5 }
      : { z: Z.dense, width: 12.2, height: 7.4, margin: 1.12, centerY: 0 },
    { z: Z.answer, width: 4.8, height: 4.8, margin: 1.35, centerY: 0 },
  ];
}

/**
 * The part of the frame the scene is allowed to use.
 *
 * The story panel is a fixed slab of the viewport, but the camera was framing content
 * against the whole of it, so the two were laid out in ignorance of each other and
 * collided whenever the content was wide. Reserving the gutter here fixes every stage at
 * once, which no amount of nudging individual labels could do.
 *
 * On desktop the panel is on the left, so the reserve is horizontal. In portrait it moves
 * to the bottom, so the reserve is vertical instead.
 */
export function safeArea(aspect: number): {
  width: number;
  height: number;
  /** Centre of the usable band, as a fraction of the viewport from the top. */
  centerY: number;
} {
  // Portrait has chrome at BOTH ends, and the band between them is not centred.
  //
  // Measured on a 390x806 viewport: the topbar ends at 68 and the story panel starts at
  // 601, so the usable band is 0.661 of the height with its centre at 0.415. Describing
  // it as a single centred fraction, which is what this used to return, put the content
  // at 0.36 and ran the panel headings under the brand in the topbar.
  //
  // Landscape keeps a centred band; its reserve is horizontal.
  return isPortrait(aspect)
    ? { width: 0.95, height: 0.66, centerY: 0.415 }
    : { width: 0.8, height: 0.92, centerY: 0.5 };
}

/**
 * Distance at which a box of `width` x `height` fits the usable part of the frame.
 *
 * `aspect` is the real frustum aspect and decides the horizontal fit. `layoutAspect`
 * decides which reserve applies, and is the value that agrees with the stylesheet about
 * where the story panel sits; the two differ in a near-square or small landscape window.
 */
export function fitDistance(frame: StationFrame, aspect: number, layoutAspect = aspect): number {
  const t = Math.tan(FOV / 2);
  const safe = safeArea(layoutAspect);
  const byHeight = (frame.height * frame.margin) / safe.height / 2 / t;
  const byWidth =
    (frame.width * frame.margin) / safe.width / 2 / (t * Math.max(aspect, 0.2));
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
const POOL2_GRID_LANDSCAPE: GridSpec = { cols: 4, rows: 4, cell: 1.06, plate: 0.98 };
// Smaller in portrait: it sits above the lattice rather than beside it, so its height is
// competing with everything below it.
const POOL2_GRID_PORTRAIT: GridSpec = { cols: 4, rows: 4, cell: 0.68, plate: 0.63 };
export function pool2Grid(aspect: number): GridSpec {
  return isPortrait(aspect) ? POOL2_GRID_PORTRAIT : POOL2_GRID_LANDSCAPE;
}
export const POOL2_GRID = POOL2_GRID_LANDSCAPE;

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

/*
 * The dense station has two shapes.
 *
 * In landscape it reads left to right: pooled features, then the hidden units, then the
 * candidates, with the three dot-product panels side by side. All of that is about twelve
 * world units wide against roughly seven tall.
 *
 * On a phone that is exactly backwards. A 390x806 viewport is 0.48 aspect, so fitting
 * twelve units of width means the camera solves to a visible height of about thirty for
 * content five units tall: the whole station shrinks into a thin band with two thirds of
 * the screen empty, and the three panel headings, which are fixed-size DOM text, run into
 * each other and off the left edge.
 *
 * So portrait turns the station through ninety degrees. The same objects, stacked down
 * the screen instead of across it, sized to the axis that actually has room.
 */

export const HIDDEN_X = -0.35;

export function hiddenGrid(): { cols: number; rows: number } {
  // Four by eight in both orientations. A wide-and-short lattice was the obvious portrait
  // move and it was wrong: it forces the candidates underneath, and three groups stacked
  // vertically do not fit between the topbar and the story panel. Keeping the lattice
  // narrow lets it sit beside the candidates instead, which is what the horizontal axis
  // is free for.
  return { cols: 4, rows: 8 };
}

/** Centre of the lattice block. */
export function hiddenOrigin(aspect: number): [number, number] {
  return isPortrait(aspect) ? [-1.4, -1.2] : [HIDDEN_X, 0];
}

export function hiddenSlot(i: number, z: number, aspect: number): Vec3 {
  const { cols, rows } = hiddenGrid();
  const col = i % cols;
  const row = Math.floor(i / cols);
  const [ox, oy] = hiddenOrigin(aspect);
  return [ox + (col - (cols - 1) / 2) * 0.46, oy - (row - (rows - 1) / 2) * 0.58, z];
}

export const POOL2_BLOCK_X = -4.3;

export function pool2Origin(aspect: number): [number, number] {
  return isPortrait(aspect) ? [0, 3.3] : [POOL2_BLOCK_X, 0];
}

// -- The dense layer, drawn as a dot product ---------------------------------

/**
 * Three panels, all in the same 16 x 7x7 layout: what one hidden unit is looking for,
 * what the drawing actually has, and where the two agree.
 *
 * A hidden unit's 784 weights come back in the same channel-major order as the pooled
 * features, so they reshape onto exactly this grid. That is what lets the dot product be
 * drawn rather than asserted, and it is the only reason a dense layer can be made
 * legible at all.
 */
const PANEL_GRID_LANDSCAPE: GridSpec = { cols: 4, rows: 4, cell: 0.78, plate: 0.72 };
/**
 * Slightly smaller in portrait, because three of them stack.
 *
 * At the landscape size the stack spanned 0.734 of the viewport against a usable band of
 * 0.69, so the first heading sat above the topbar and the running total below the story
 * panel. Eleven percent off the cell buys exactly the room needed and the 7x7 maps are
 * still legible at it.
 */
const PANEL_GRID_PORTRAIT: GridSpec = { cols: 4, rows: 4, cell: 0.68, plate: 0.63 };

export function panelGrid(aspect: number): GridSpec {
  return isPortrait(aspect) ? PANEL_GRID_PORTRAIT : PANEL_GRID_LANDSCAPE;
}

/** Landscape shape, for the places that only need nominal sizes. */
export const PANEL_GRID = PANEL_GRID_LANDSCAPE;
export const PANEL_X = [-3.6, 0, 3.6] as const;
/** The panels sit above centre so the running total below them has room. */
export const PANEL_Y = 0.5;
/**
 * Portrait stacks the three panels down the screen.
 *
 * Pitch is 3.45: a panel is 3.12 tall and its heading takes the rest. Side by side they
 * would be 390px apart on a 390px screen, so all three headings collide and the first
 * runs off the left edge.
 *
 * The pitch is what it is because the stack has to fit between the top of the frame and
 * the top of the story panel, measured at 0.746 of the viewport on a 390x806 screen. At
 * 3.9 the first heading projected to -0.038, above the frame, and the running total to
 * 0.753, underneath the panel.
 */
const PANEL_Y_PORTRAIT = [3.05, 0, -3.05] as const;

/** Centre of a panel, in whichever arrangement is in play. */
export function panelOrigin(panel: number, aspect: number): [number, number] {
  return isPortrait(aspect) ? [0, PANEL_Y_PORTRAIT[panel]] : [PANEL_X[panel], PANEL_Y];
}

// Clear of the panel top, which reaches its origin + 1.56 + half a plate.
export function panelLabelY(panel: number, aspect: number): number {
  return panelOrigin(panel, aspect)[1] + (isPortrait(aspect) ? 1.58 : 1.82);
}

/** Where the agreement collapses to a single running total. */
export function sumY(aspect: number): number {
  // Below the lowest panel in each arrangement.
  return isPortrait(aspect) ? -5.15 : -2.1;
}
export const SUM_MAX_WIDTH = 3.4;

/**
 * Where the running total starts.
 *
 * Zero in landscape, because the bar growing rightwards from a zero line under the middle
 * panel is the picture. In portrait the panels are a centred column, so a bar starting at
 * zero would hang off their right-hand side and out of frame; it starts half a span left
 * instead, which keeps the same reading and stays under the stack.
 */
export function sumOriginX(aspect: number): number {
  return isPortrait(aspect) ? -SUM_MAX_WIDTH / 2 : 0;
}

export function panelSlot(panel: number, i: number, z: number, aspect: number): Vec3 {
  const c = gridCell(panelGrid(aspect), i, z);
  const [ox, oy] = panelOrigin(panel, aspect);
  return [c[0] + ox, c[1] + oy, c[2]];
}

export function pool2Slot(i: number, z: number, aspect: number): Vec3 {
  const c = gridCell(pool2Grid(aspect), i, z);
  const [ox, oy] = pool2Origin(aspect);
  return [c[0] + ox, c[1] + oy, c[2]];
}

export const CANDIDATE_X = 4.05;

/** Where the column of candidate digits sits, and how far apart they are. */
export function candidateLayout(aspect: number): { x: number; pitch: number } {
  // Tighter and centred in portrait: ten at the landscape pitch is 6.9 units tall, which
  // does not fit under a lattice that has already used the vertical axis.
  return isPortrait(aspect) ? { x: 1.65, pitch: 0.46 } : { x: CANDIDATE_X, pitch: 0.685 };
}

/** Zero line of the for/against meter that sits beside each candidate. */
export function railAxisX(aspect: number): number {
  return candidateLayout(aspect).x - (isPortrait(aspect) ? 0.95 : 1.42);
}

/**
 * The line the station's explanatory tags sit on.
 *
 * Below the lowest candidate rail and still inside the frame. In landscape the station is
 * framed on its width, which leaves the visible half-height at about 4.5 against content
 * that stops at 3.2; portrait is framed on its height, so the line has to sit under a
 * taller stack.
 */
export function floorLabelY(aspect: number): number {
  return isPortrait(aspect) ? -4.25 : -3.85;
}

/** The ten candidate digits, stacked vertically while the network is deciding. */
export function candidateSlot(digit: number, z: number, aspect: number): Vec3 {
  const { x, pitch } = candidateLayout(aspect);
  const y = -(digit - 4.5) * pitch;
  return [x, isPortrait(aspect) ? y - 1.05 : y, z];
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

/*
 * Portrait turns the chart on its side.
 *
 * Ten categories across a 390px screen is 39 CSS pixels a column, and the value labels
 * are numbers like "-21.5" that measure about 34. They cannot not collide, at any type
 * size that is still readable. Rows put the ten categories on the axis that has room and
 * leave a whole column free for the digit and another for its value, which is the
 * ordinary responsive treatment for a bar chart and reads better than the landscape one
 * on a phone.
 */
const ROW_PITCH = 0.95;
const ROW_THICK = 0.56;
const ROW_AXIS_X = 0.55;
const ROW_MAX_LEN = 2.25;
/** The prototype glyph and the numeral each get their own column, left of the axis. */
export const ROW_PROTO_X = ROW_AXIS_X - 3.35;
const ROW_DIGIT_X = ROW_AXIS_X - 2.35;

export interface BarPlacement {
  centre: [number, number];
  size: [number, number];
  digit: [number, number];
  value: [number, number];
  /** Full-scale length, for normalising intensity the same way in both orientations. */
  span: number;
}

/**
 * Where one class's bar, its numeral and its reading sit.
 *
 * `h` is the signed magnitude on the landscape scale, so every caller keeps computing the
 * same number and only the placement changes.
 */
export function barPlacement(digit: number, h: number, aspect: number): BarPlacement {
  if (!isPortrait(aspect)) {
    return {
      centre: [barX(digit), BAR_BASE_Y + h / 2],
      size: [BAR_WIDTH, Math.abs(h) + 0.02],
      // Diverging bars, so the numeral takes whichever side of the axis its own bar is
      // not using. 0.52 rather than 0.34 because a bar is a bloomed sprite and the
      // winner is the brightest thing on screen.
      digit: [barX(digit), BAR_BASE_Y + (h >= 0 ? -0.52 : 0.52)],
      value: [barX(digit), BAR_BASE_Y + h + (h >= 0 ? 0.5 : -0.5)],
      span: BAR_MAX_HEIGHT,
    };
  }

  const len = (h / BAR_MAX_HEIGHT) * ROW_MAX_LEN;
  const y = -(digit - 4.5) * ROW_PITCH;
  return {
    centre: [ROW_AXIS_X + len / 2, y],
    size: [Math.abs(len) + 0.02, ROW_THICK],
    // A fixed column, because in rows there is no "free side" to alternate onto: the
    // numeral would otherwise land on its neighbour's bar.
    digit: [ROW_DIGIT_X, y],
    value: [ROW_AXIS_X + len + (len >= 0 ? 0.52 : -0.52), y],
    span: ROW_MAX_LEN,
  };
}

/** The zero line the bars diverge from, and how long it runs. */
export function barAxis(aspect: number): { centre: [number, number]; size: [number, number] } {
  return isPortrait(aspect)
    ? { centre: [ROW_AXIS_X, 0], size: [0.012, CLASSES_SPAN * ROW_PITCH + 0.6] }
    : { centre: [0, BAR_BASE_Y], size: [CLASSES_SPAN * BAR_SPACING + 0.2, 0.012] };
}

/** The container softmax divides: one unit of certainty, drawn as an object. */
export function budgetBox(aspect: number): { centre: [number, number]; size: [number, number] } {
  return isPortrait(aspect)
    ? {
        centre: [ROW_AXIS_X + ROW_MAX_LEN * 0.5, 0],
        size: [ROW_MAX_LEN + 0.35, CLASSES_SPAN * ROW_PITCH + 0.75],
      }
    : {
        centre: [0, BAR_BASE_Y + BAR_MAX_HEIGHT * 0.5],
        size: [CLASSES_SPAN * BAR_SPACING + 0.35, BAR_MAX_HEIGHT + 0.3],
      };
}

/** Where the "raw score / exponentiate / one unit" reading sits. */
export function readingAnchor(aspect: number): [number, number] {
  return isPortrait(aspect) ? [ROW_AXIS_X - 0.6, CLASSES_SPAN * ROW_PITCH * 0.5 + 0.45] : [0, BAR_MAX_HEIGHT + 0.4];
}

/** Where a candidate prototype flies to when the bars gather. */
export function protoTarget(digit: number, aspect: number): [number, number] {
  return isPortrait(aspect)
    ? [ROW_PROTO_X, -(digit - 4.5) * ROW_PITCH]
    : [barX(digit), BAR_BASE_Y + BAR_MAX_HEIGHT + 1.3];
}

const CLASSES_SPAN = 10;

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
