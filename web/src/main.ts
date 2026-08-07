/**
 * Application shell. Owns the render loop, the mode machine and the wiring between the
 * DOM and the scene. Deliberately the only place that reaches for `document`.
 */

import './styles.css';
import { Audio, type CueName } from './audio/audio';
import { Engine, type Run } from './core/engine';
import { Player, type Sampled } from './core/timeline';
import { buildScore, stageAt, type Score } from './scene/choreography';
import { CANVAS_RES } from './scene/constants';
import { Scene } from './scene/scene';
import { DrawSurface } from './ui/draw';
import { explain, formatConfidence, verdict } from './ui/copy';
import { AnnotationLayer } from './ui/annotations';

type Mode = 'loading' | 'compose' | 'reveal' | 'result';

const SPEEDS = [0.5, 1, 1.5, 2];

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

class App {
  private engine!: Engine;
  scene!: Scene;
  surface!: DrawSurface;
  private audio = new Audio();
  private annotations!: AnnotationLayer;
  player = new Player();
  score: Score | null = null;
  run: Run | null = null;
  private sampled: Sampled = {};

  private mode: Mode = 'loading';
  private speedIndex = 1;
  private lastFrame = 0;
  private scrubbing = false;
  private recorder: MediaRecorder | null = null;
  private reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  async boot() {
    const stage = $<HTMLCanvasElement>('stage');
    this.surface = new DrawSurface($<HTMLCanvasElement>('pad'));
    this.annotations = new AnnotationLayer($('annos'));

    try {
      this.engine = await Engine.load('./model/');
      this.scene = new Scene(stage, this.engine);
    } catch (err) {
      // The full detail (including shader source on a compile failure) belongs in the
      // console, not on screen.
      console.error('[p2p] startup failed', err);
      $('boot').hidden = true;
      $('unsupported').hidden = false;
      const detail = err instanceof Error ? err.message.split('\n')[0] : String(err);
      $('unsupported').querySelector('p')!.textContent =
        `${detail}. The full error is in the browser console.`;
      return;
    }

    $('modelBadge').textContent = `${this.engine.info.params.toLocaleString()} parameters · ${(
      this.engine.info.testAccuracy * 100
    ).toFixed(2)}% on MNIST · runs on your device`;

    this.wire();
    this.resize();
    this.setMode('compose');
    $('app').hidden = false;

    // Development hook so the stage-by-stage audit can seek precisely rather than
    // clicking through the transport and hoping.
    if (import.meta.env.DEV) {
      (window as unknown as Record<string, unknown>).__p2p = this;
    }

    requestAnimationFrame(this.loop);
  }

  // ---------------------------------------------------------------- wiring

  private wire() {
    this.surface.onChange = () => {
      const inked = !this.surface.isEmpty;
      $('pad').parentElement?.classList.toggle('pad--inked', inked);
      document.querySelector('.pad')?.classList.toggle('pad--inked', inked);
      $<HTMLButtonElement>('revealBtn').disabled = !inked;
    };

    $('revealBtn').addEventListener('click', () => void this.reveal());
    $('clearBtn').addEventListener('click', () => {
      this.surface.clear();
      $('composeFoot').textContent = '';
    });
    $('undoBtn').addEventListener('click', () => this.surface.undo());

    $('uploadInput').addEventListener('change', (e) => {
      const input = e.target as HTMLInputElement;
      const file = input.files?.[0];
      // Reset the value so re-picking the same file still fires a change event. Without
      // this, clearing and re-uploading the same photo appears to do nothing.
      input.value = '';
      if (file) {
        void this.surface
          .loadImage(file)
          .catch(() => this.toast('Could not read that image'));
      }
    });

    $('againBtn').addEventListener('click', () => {
      this.surface.clear();
      this.player.pause();
      // Drop the run as well as the timeline, otherwise the previous answer stays
      // painted on the canvas behind the drawing panel.
      this.score = null;
      this.run = null;
      this.lastStageKey = '';
      this.scene.clearRun();
      // Quiet the drone while nobody is watching a run.
      this.audio.fadePad(0);
      $('composeFoot').textContent = '';
      this.setMode('compose');
    });
    $('replayBtn').addEventListener('click', () => {
      this.setMode('reveal');
      this.player.restart();
    });
    $('shareBtn').addEventListener('click', () => void this.record());

    $('playBtn').addEventListener('click', () => this.player.toggle());
    $('prevBtn').addEventListener('click', () => this.jumpStage(-1));
    $('nextBtn').addEventListener('click', () => this.jumpStage(1));
    $('skipBtn').addEventListener('click', () => {
      if (this.score) this.player.seek(this.score.timeline.duration);
    });

    $('speedBtn').addEventListener('click', () => {
      this.speedIndex = (this.speedIndex + 1) % SPEEDS.length;
      this.player.speed = SPEEDS[this.speedIndex];
      $('speedBtn').textContent = `${SPEEDS[this.speedIndex]}×`;
    });

    const sound = $<HTMLButtonElement>('soundBtn');
    sound.setAttribute('aria-pressed', String(this.audio.enabled));
    sound.addEventListener('click', () => {
      this.audio.setEnabled(!this.audio.enabled);
      sound.setAttribute('aria-pressed', String(this.audio.enabled));
    });

    this.wireScrub();

    this.player.onCue = (cue) => this.audio.play(cue.name as CueName);
    this.player.onStateChange = (state) => {
      $('playIcon').textContent = state === 'playing' ? '❚❚' : '▶';
      if (state === 'finished') this.setMode('result');
    };

    window.addEventListener('resize', () => this.resize());
    window.addEventListener('orientationchange', () => this.resize());

    window.addEventListener('pointermove', (e) => {
      // Gentle parallax. Decorative only, and never applied while recording.
      this.scene.parallax = [
        (e.clientX / window.innerWidth - 0.5) * 2,
        -(e.clientY / window.innerHeight - 0.5) * 2,
      ];
    });

    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      // A focused control handles its own keys. Without this, Enter on the reveal button
      // and Space on the play button fire twice (once from the element, once from here),
      // which starts two runs or toggles play straight back off.
      if (e.target instanceof HTMLElement && e.target.closest('button, [role="slider"], a')) {
        return;
      }
      if (e.code === 'Space' && this.mode !== 'compose') {
        e.preventDefault();
        this.player.toggle();
      } else if (e.code === 'ArrowRight') {
        this.jumpStage(1);
      } else if (e.code === 'ArrowLeft') {
        this.jumpStage(-1);
      } else if (e.code === 'Enter' && this.mode === 'compose' && !this.surface.isEmpty) {
        void this.reveal();
      } else if (e.key.toLowerCase() === 'c') {
        this.surface.clear();
      }
    });
  }

  private wireScrub() {
    const scrub = $('scrub');
    const seekTo = (clientX: number) => {
      if (!this.score) return;
      const r = scrub.getBoundingClientRect();
      const t = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      this.player.seek(t * this.score.timeline.duration);
    };

    scrub.addEventListener('pointerdown', (e) => {
      this.scrubbing = true;
      scrub.setPointerCapture(e.pointerId);
      this.player.pause();
      seekTo(e.clientX);
    });
    scrub.addEventListener('pointermove', (e) => {
      if (this.scrubbing) seekTo(e.clientX);
    });
    const stop = () => {
      this.scrubbing = false;
    };
    scrub.addEventListener('pointerup', stop);
    scrub.addEventListener('pointercancel', stop);

    scrub.addEventListener('keydown', (e) => {
      if (!this.score) return;
      const step = this.score.timeline.duration / 40;
      if (e.key === 'ArrowRight') this.player.seek(this.player.time + step);
      else if (e.key === 'ArrowLeft') this.player.seek(this.player.time - step);
      else return;
      e.preventDefault();
    });
  }

  // ----------------------------------------------------------------- modes

  private setMode(mode: Mode) {
    this.mode = mode;
    document.body.dataset.mode = mode;
  }

  private revealing = false;

  private async reveal() {
    // `reveal` awaits before it can disable anything, so without a guard a double click
    // (or Enter, which fires both the button handler and the window handler) starts two
    // runs: two forward passes, two scenes uploaded, and the animation snapping back to
    // zero mid-play.
    if (this.revealing || this.mode !== 'compose') return;
    this.revealing = true;
    try {
      await this.startRun();
    } catch (err) {
      console.error('[p2p] reveal failed', err);
      this.toast('Something went wrong reading that drawing');
    } finally {
      this.revealing = false;
    }
  }

  private async startRun() {
    await this.audio.ensure();
    this.audio.startPad();
    this.audio.fadePad(0.13);

    const source = this.surface.extract();
    const run = this.engine.run(source, CANVAS_RES, CANVAS_RES);
    if (!run) {
      this.toast('Nothing to read there yet');
      return;
    }

    this.run = run;
    this.scene.setRun(run, source);
    this.score = buildScore(run);
    this.renderMarks(this.score);
    this.fillResult(run);

    $('composeFoot').textContent = `forward pass and gradients in ${run.ms.toFixed(2)} ms`;

    this.setMode('reveal');
    this.player.load(this.score.timeline, true);

    // Someone who asked not to be animated still deserves the answer.
    if (this.reducedMotion) {
      this.player.seek(this.score.timeline.duration);
      this.player.pause();
      this.setMode('result');
    }
  }

  private jumpStage(dir: number) {
    if (!this.score) return;
    const stages = this.score.stages;
    const current = stageAt(stages, this.player.time + 0.12);
    let index = current.index + dir;
    // Stepping back near the start of a stage should return to its beginning first.
    if (dir < 0 && this.player.time - current.start > 0.9) index = current.index;
    index = Math.min(stages.length - 1, Math.max(0, index));
    this.player.seek(stages[index].start);
    if (this.mode === 'result') this.setMode('reveal');
  }

  private renderMarks(score: Score) {
    const marks = $('scrubMarks');
    marks.innerHTML = '';
    for (const stage of score.stages) {
      if (stage.index === 0) continue;
      const el = document.createElement('div');
      el.className = 'scrub__mark';
      el.style.left = `${(stage.start / score.timeline.duration) * 100}%`;
      marks.appendChild(el);
    }
  }

  private fillResult(run: Run) {
    const e = run.evidence;
    const words = explain(e);

    $('resultDigit').textContent = String(e.top1);
    $('resultPct').textContent = formatConfidence(e.p1);
    $('resultPhrase').textContent = `${verdict(e)} ${words.confidence}.`;
    $('resultReason').textContent = words.reason;
    $('resultCounter').textContent = words.counter;

    const dist = $('dist');
    dist.innerHTML = '';
    for (let d = 0; d < 10; d++) {
      const col = document.createElement('div');
      col.className = 'dist__col' + (d === e.top1 ? ' dist__col--top' : '');
      const bar = document.createElement('div');
      bar.className = 'dist__bar';
      // A square root makes the small probabilities visible without lying about which
      // one won; the label carries the exact number.
      bar.style.height = `${Math.max(2, Math.sqrt(run.probs[d]) * 100)}%`;
      bar.title = `${d}: ${(run.probs[d] * 100).toFixed(2)}%`;
      const label = document.createElement('span');
      label.className = 'dist__label';
      label.textContent = String(d);
      col.append(bar, label);
      dist.appendChild(col);
    }
  }

  // ------------------------------------------------------------- recording

  private async record() {
    if (!this.score || this.recorder) return;
    const canvas = $<HTMLCanvasElement>('stage');
    if (typeof canvas.captureStream !== 'function' || typeof MediaRecorder === 'undefined') {
      this.toast('Recording is not supported in this browser');
      return;
    }

    const stream = canvas.captureStream(60);
    const audioStream = this.audio.captureStream();
    if (audioStream) for (const track of audioStream.getAudioTracks()) stream.addTrack(track);

    const mime = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'].find(
      (m) => MediaRecorder.isTypeSupported(m),
    );
    if (!mime) {
      this.toast('Recording is not supported in this browser');
      return;
    }

    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = () => {
      // Stop every track. A canvas capture track keeps copying each rendered frame for
      // the lifetime of the page otherwise, so one Record click permanently taxes every
      // frame after it.
      for (const track of stream.getTracks()) track.stop();
      this.audio.releaseStream();

      const blob = new Blob(chunks, { type: mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pixel-to-prediction-${this.run?.evidence.top1 ?? 'digit'}.webm`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      this.recorder = null;
      this.scene.parallaxEnabled = true;
      this.toast('Clip saved');
    };

    // Freeze the parallax so the recording is reproducible rather than dependent on
    // where the pointer happened to be.
    this.scene.parallaxEnabled = false;
    this.scene.parallax = [0, 0];
    this.recorder = recorder;
    recorder.start();
    this.toast('Recording…');

    this.setMode('reveal');
    this.player.speed = 1;
    this.speedIndex = 1;
    $('speedBtn').textContent = '1×';
    this.player.restart();

    const total = this.score.timeline.duration * 1000 + 1400;
    setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, total);
  }

  private toastTimer = 0;
  private toast(message: string) {
    const el = $('toast');
    el.textContent = message;
    el.classList.add('toast--on');
    clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => el.classList.remove('toast--on'), 2600);
  }

  // ----------------------------------------------------------------- frame

  private resize() {
    this.scene.renderer.resize(window.innerWidth, window.innerHeight);
  }

  /**
   * Render one exact timeline position and hand back a PNG.
   *
   * The draw and the read happen in the same task on purpose: the context is created
   * without preserveDrawingBuffer, so the buffer is only valid until the browser
   * composites. Reading a frame later would return transparent black.
   */
  renderFrameAt(time: number): string {
    if (!this.score) return '';
    this.player.pause();
    this.player.seek(time);
    const t = this.player.time;
    this.score.timeline.sample(t, this.sampled);
    this.scene.parallaxEnabled = false;
    this.scene.parallax = [0, 0];
    this.scene.draw(this.sampled, t, 16.7);
    this.annotations.render(this.scene.labels);
    this.updateChrome(t);
    return $<HTMLCanvasElement>('stage').toDataURL('image/png');
  }

  /** Dump a set of timeline positions to disk through the dev server. */
  async dumpFrames(times: number[], prefix = 'stage'): Promise<string[]> {
    const written: string[] = [];
    for (const t of times) {
      const dataUrl = this.renderFrameAt(t);
      if (!dataUrl) continue;
      const name = `${prefix}-${t.toFixed(2).replace('.', '_')}.png`;
      const res = await fetch('/__frame', {
        method: 'POST',
        body: JSON.stringify({ name, dataUrl }),
      });
      written.push(`${name}:${res.ok ? 'ok' : 'fail'}`);
    }
    return written;
  }

  private loop = (now: number) => {
    requestAnimationFrame(this.loop);
    const frameMs = this.lastFrame ? Math.min(64, now - this.lastFrame) : 16.7;
    this.lastFrame = now;

    if (this.score) {
      const t = this.player.advance(frameMs / 1000);
      this.score.timeline.sample(t, this.sampled);
      this.scene.draw(this.sampled, t, frameMs);
      this.annotations.render(this.scene.labels);
      this.updateChrome(t);
    } else {
      this.scene.draw(this.sampled, now / 1000, frameMs);
      this.annotations.clear();
    }
  };

  private lastStageKey = '';
  private updateChrome(t: number) {
    if (!this.score) return;
    const stage = stageAt(this.score.stages, t);
    if (stage.key !== this.lastStageKey) {
      this.lastStageKey = stage.key;
      $('storyStep').textContent = `${stage.index + 1} / ${this.score.stages.length}`;
      $('storyTitle').textContent = stage.title;
      $('storyCaption').textContent = stage.caption;
    }

    const pct = this.player.progress * 100;
    $('scrubFill').style.width = `${pct}%`;
    $('scrub').setAttribute('aria-valuenow', String(Math.round(pct)));

    // The answer panel belongs with the final stage, not with the end of the timeline.
    // Waiting for the last frame left the prediction and confidence hidden for several
    // seconds after the network had visibly committed to them.
    const final = this.score.stages[this.score.stages.length - 1];
    const revealAt = final.start + 1.15;
    if (t >= revealAt && this.mode !== 'result') this.setMode('result');
    else if (t < revealAt && this.mode === 'result') this.setMode('reveal');
  }
}

const app = new App();
void app.boot();
