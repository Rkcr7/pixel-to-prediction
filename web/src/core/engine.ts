/**
 * Loads the WASM core and turns one drawing into one complete `Run`: every tensor the
 * visualisation is permitted to draw from, captured in a single forward pass plus two
 * backward passes.
 */

import init, { Engine as WasmEngine, decode_prototypes } from '../wasm/nnviz';

export interface Evidence {
  top1: number;
  top2: number;
  p1: number;
  p2: number;
  margin: number;
  holes: number;
  inkRatio: number;
  regions: number[];
  counterRegions: number[];
  strongestRegion: number;
  contestedRegion: number;
  diffuse: boolean;
}

export interface Run {
  input: Float32Array;
  conv1Pre: Float32Array;
  conv1: Float32Array;
  pool1: Float32Array;
  pool1Winners: Uint8Array;
  conv2Pre: Float32Array;
  conv2: Float32Array;
  pool2: Float32Array;
  pool2Winners: Uint8Array;
  fc1Pre: Float32Array;
  fc1: Float32Array;
  logits: Float32Array;
  exps: Float32Array;
  probs: Float32Array;
  conv1Energy: Float32Array;
  conv2Energy: Float32Array;
  fc2Contributions: Float32Array;
  fc1TopContributions: Float32Array;
  saliency: Float32Array;
  counterfactual: Float32Array;
  evidence: Evidence;
  /** Class indices sorted by probability, strongest first. */
  ranked: number[];
  /** Channel indices sorted by activation energy, strongest first. */
  conv1Order: number[];
  conv2Order: number[];
  prep: {
    bbox: [number, number, number, number];
    boxed: Float32Array;
    boxedW: number;
    boxedH: number;
    com: [number, number];
    offset: [number, number];
  };
  /** Wall-clock milliseconds for the whole forward + explanation pass. */
  ms: number;
}

export interface ModelInfo {
  params: number;
  testAccuracy: number;
}

const rankDesc = (values: ArrayLike<number>): number[] =>
  Array.from({ length: values.length }, (_, i) => i).sort((a, b) => values[b] - values[a]);

export class Engine {
  private wasm!: WasmEngine;
  /** Ten 28x28 class prototypes: the mean of every training image of that digit. */
  prototypes!: Float32Array;
  kernels1!: Float32Array;
  kernels2!: Float32Array;
  info: ModelInfo = { params: 0, testAccuracy: 0 };

  static async load(baseUrl = './model/'): Promise<Engine> {
    const engine = new Engine();
    const [, modelBytes, protoBytes, meta] = await Promise.all([
      init(),
      fetchBytes(`${baseUrl}model.bin`),
      fetchBytes(`${baseUrl}prototypes.bin`),
      fetch(`${baseUrl}model.json`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null),
    ]);

    engine.wasm = new WasmEngine(modelBytes);
    engine.prototypes = decode_prototypes(protoBytes);
    engine.kernels1 = engine.wasm.kernels1();
    engine.kernels2 = engine.wasm.kernels2();
    engine.info = {
      params: engine.wasm.params,
      testAccuracy: meta?.test_accuracy ?? 0,
    };
    return engine;
  }

  /**
   * `source` is a square grayscale buffer with ink as high values in [0,1].
   * Returns null when there is no ink to classify.
   */
  run(source: Float32Array, width: number, height: number): Run | null {
    const t0 = performance.now();
    const ok = this.wasm.run(source, width, height);
    if (!ok) return null;

    const w = this.wasm;
    const probs = w.probs();
    const conv1Energy = w.conv1_energy();
    const conv2Energy = w.conv2_energy();
    const boxedSize = w.boxed_size();
    const centring = w.centring();
    const bbox = w.bbox();

    const run: Run = {
      input: w.input(),
      conv1Pre: w.conv1_pre(),
      conv1: w.conv1(),
      pool1: w.pool1(),
      pool1Winners: w.pool1_winners(),
      conv2Pre: w.conv2_pre(),
      conv2: w.conv2(),
      pool2: w.pool2(),
      pool2Winners: w.pool2_winners(),
      fc1Pre: w.fc1_pre(),
      fc1: w.fc1(),
      logits: w.logits(),
      exps: w.exps(),
      probs,
      conv1Energy,
      conv2Energy,
      fc2Contributions: w.fc2_contributions(),
      fc1TopContributions: w.fc1_top_contributions(10),
      saliency: w.saliency(),
      counterfactual: w.counterfactual(),
      evidence: JSON.parse(w.evidence()) as Evidence,
      ranked: rankDesc(probs),
      conv1Order: rankDesc(conv1Energy),
      conv2Order: rankDesc(conv2Energy),
      prep: {
        bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
        boxed: w.boxed(),
        boxedW: boxedSize[0],
        boxedH: boxedSize[1],
        com: [centring[0], centring[1]],
        offset: [centring[2], centring[3]],
      },
      ms: 0,
    };
    run.ms = performance.now() - t0;
    return run;
  }

  /** Gradient ascent on a blank canvas: what this network thinks a digit looks like. */
  dream(digit: number, steps = 140): Float32Array {
    return this.wasm.dream(digit, steps);
  }

  /** One prototype as a 784-length view. */
  prototype(digit: number): Float32Array {
    return this.prototypes.subarray(digit * 784, (digit + 1) * 784);
  }
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}: ${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
}
