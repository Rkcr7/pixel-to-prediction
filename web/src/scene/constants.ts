/** Network shape, mirrored from the Rust core. Changing one without the other is a bug. */

export const IMG = 28;
export const C1 = 8; // conv1 filters
export const K1 = 5; // conv1 kernel size
export const C2 = 16; // conv2 filters
export const K2 = 3;
export const POOL1 = 14;
export const POOL2 = 7;
export const HIDDEN = 32;
export const CLASSES = 10;

/** Resolution of the drawing canvas buffer handed to the preprocessor. */
export const CANVAS_RES = 224;
