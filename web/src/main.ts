/**
 * Application shell. Owns the render loop, the mode machine and the wiring between the
 * DOM and the scene. Deliberately the only place that reaches for `document`.
 */

import './styles.css';
import { Audio, type CueName } from './audio/audio';
import { Engine, type Run } from './core/engine';
import { Player, type Sampled } from './core/timeline';
import { buildScore, NOTES, stageAt, type Score } from './scene/choreography';
import { CANVAS_RES } from './scene/constants';
import { Scene } from './scene/scene';
import { DrawSurface } from './ui/draw';
import { explain, formatConfidence, verdict } from './ui/copy';
import { AnnotationLayer } from './ui/annotations';
import { FrameCompositor } from './ui/compositor';

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
  /** Set once the GPU context is gone. Nothing may touch GL after this. */
  private contextLost = false;
  /**
   * The challenge in play, if any: what the network said last time, and which class the
   * hint on the pad is pushing towards. Null whenever the user is just drawing.
   */
  private fooling: { from: number; to: number } | null = null;
  private recorder: MediaRecorder | null = null;
  /**
   * The recorded frame, scene and text together.
   *
   * Only painted while a recording is running: it is a full extra canvas blit plus text
   * every frame, and nothing watches it the rest of the time.
   */
  private compositor = new FrameCompositor();
  private recording = false;
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
    this.wireContextLoss();
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
      this.fooling = null;
      this.returnToPad();
    });

    // Back to the pad with the drawing kept and the flip hint painted under it.
    //
    // Keeping the strokes is the whole feature. "Try another digit" starts over; this
    // one says "that drawing, one stroke different", which is the only version of the
    // question the hint can answer, since the hint is a first-order statement about
    // exactly this input.
    $('foolBtn').addEventListener('click', () => {
      const run = this.run;
      if (!run) return;
      this.fooling = { from: run.evidence.top1, to: run.evidence.top2 };
      this.surface.setHint(run.flipHint, {
        bbox: run.prep.bbox,
        boxedW: run.prep.boxedW,
        boxedH: run.prep.boxedH,
        offset: run.prep.offset,
      });
      this.returnToPad();
      $('composeFoot').textContent =
        `It said ${this.fooling.from}. Add ink where the pad glows to push it towards ${this.fooling.to}.`;
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

    // Wake the transport on any deliberate input, then let it go again.
    for (const type of ['pointermove', 'pointerdown', 'keydown', 'wheel'] as const) {
      window.addEventListener(type, () => this.wakeChrome(), { passive: true });
    }
    // Keep it up while the pointer is actually over it, so it cannot vanish mid-reach.
    $('transport').addEventListener('pointerenter', () => this.wakeChrome(9_000));

    window.addEventListener('pointermove', (e) => {
      // Gentle parallax. Decorative only, and never applied while recording.
      //
      // Mouse only. A touch drag also emits pointermove, so on a phone the scene lurched
      // sideways every time someone swiped or scrolled, and then stayed there: there is
      // no pointerleave on touch to put it back. A decorative effect that needs a hover
      // position has no meaning on a device with no hover.
      if (e.pointerType !== 'mouse') return;
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
    // Entering a mode is deliberate input, so the controls should be there for it.
    this.wakeChrome(mode === 'reveal' ? 2600 : 9_000);
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
    // Never let the audio gate hold the run.
    //
    // `ensure` resumes an AudioContext, which needs a user gesture. With one it settles
    // immediately; without one it can simply never settle, and since `reveal` only clears
    // its re-entrancy guard in a finally block, a hung resume would wedge the button for
    // the rest of the session with nothing on screen to explain it. Silence is a far
    // smaller failure than an app that stops responding.
    await Promise.race([
      this.audio.ensure(),
      new Promise<void>((resolve) => setTimeout(resolve, 1200)),
    ]);
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

  /**
   * Back to the drawing pad, leaving whatever is on the surface alone.
   *
   * The run and the score are dropped rather than kept, because the previous answer is
   * still painted on the GL canvas that sits behind the drawing panel and would show
   * through it.
   */
  private returnToPad() {
    this.player.pause();
    this.score = null;
    this.run = null;
    this.lastStageKey = '';
    this.scene.clearRun();
    // Quiet the drone while nobody is watching a run.
    this.audio.fadePad(0);
    $('composeFoot').textContent = '';
    this.setMode('compose');
  }

  /**
   * The verdict on a challenge, if one is in play.
   *
   * "Fooled" means only that the answer moved off what it was, which is the honest claim
   * this app can make: whether a human still reads the drawing as the original digit is
   * not something the network knows, and pretending to check it would be a lie. So the
   * line reports the move and leaves the judging to the person who drew it.
   */
  private fillFoolVerdict(e: Run['evidence']) {
    const el = $('resultFool');
    const game = this.fooling;
    if (!game) {
      el.hidden = true;
      el.classList.remove('result__fool--won');
      return;
    }

    const moved = e.top1 !== game.from;
    el.classList.toggle('result__fool--won', moved);
    el.textContent = moved
      ? `You fooled it. It said ${game.from} before, and ${e.top1} now.`
      : `Still ${e.top1}. The runner-up is ${e.top2} now, so the glow has moved with it.`;
    el.hidden = false;

    // The challenge is always against the latest answer, so a failed attempt rolls
    // forward rather than chasing a target the network has already left behind.
    if (!moved) this.fooling = { from: e.top1, to: e.top2 };
    else this.fooling = null;
  }

  private fillResult(run: Run) {
    const e = run.evidence;
    const words = explain(e);

    this.fillFoolVerdict(e);
    $('resultDigit').textContent = String(e.top1);
    $('resultPct').textContent = formatConfidence(e.p1);
    // `verdict` already ends its own sentence, so the confidence phrase starts a new one
    // and has to be capitalised. Joined raw it reads "That is a 3. no doubt at all."
    const confidence = words.confidence;
    $('resultPhrase').textContent = `${verdict(e)} ${confidence
      .charAt(0)
      .toUpperCase()}${confidence.slice(1)}.`;
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

    // Capture the composited frame, not the raw scene.
    //
    // captureStream() captures one canvas and nothing else, and every word in this app is
    // DOM layered over the WebGL canvas. Recording the scene canvas produced a clip of
    // shapes moving with no annotation, no caption and no answer attached to any of it,
    // which is most of what the piece is for.
    this.compositor.sync(canvas, window.innerWidth);
    this.recording = true;
    const stream = this.compositor.canvas.captureStream(60);
    const audioStream = this.audio.captureStream();
    if (audioStream) for (const track of audioStream.getAudioTracks()) stream.addTrack(track);

    // MP4/H.264 first, WebM only as a fallback.
    //
    // This clip exists to be posted, and Instagram rejects WebM outright while LinkedIn
    // is unreliable with it — so the button was producing a file that could not be
    // uploaded to the places people would want to upload it. Chrome and Edge can encode
    // H.264 through MediaRecorder; Safari and Firefox fall through to WebM, which they
    // can, and which at least plays locally.
    const mime = [
      'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
      'video/mp4;codecs=avc1',
      'video/mp4',
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
    ].find((m) => MediaRecorder.isTypeSupported(m));
    if (!mime) {
      this.toast('Recording is not supported in this browser');
      return;
    }
    const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';

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
      a.download = `pixel-to-prediction-${this.run?.evidence.top1 ?? 'digit'}.${ext}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      this.recorder = null;
      this.recording = false;
      this.scene.parallaxEnabled = true;
      this.toast('Clip saved');
    };

    // Freeze the parallax so the recording is reproducible rather than dependent on
    // where the pointer happened to be.
    this.scene.parallaxEnabled = false;
    this.scene.parallax = [0, 0];
    this.recorder = recorder;
    recorder.start();

    // Twice speed, and the clip really is twice as fast.
    //
    // MediaRecorder timestamps frames by the wall clock they arrive on, so there is no way
    // to push 1,700 rendered frames at it faster than real time and get a correctly timed
    // video: the clip would collapse to almost no duration. Encoding offline would mean
    // WebCodecs and a container muxer. Playing the timeline faster is the honest version
    // of the same wish, and it costs a minute of waiting rather than a dependency.
    //
    // It also makes a better clip. 57 seconds is long for a feed, 29 is not, and anyone
    // who wants to read every caption can open the app and watch it at 1x.
    const speed = Math.max(2, this.player.speed);
    this.player.speed = speed;
    this.speedIndex = SPEEDS.indexOf(speed) >= 0 ? SPEEDS.indexOf(speed) : this.speedIndex;
    $('speedBtn').textContent = `${speed}×`;
    this.setMode('reveal');
    this.player.restart();

    const total = (this.score.timeline.duration / speed) * 1000 + 1400;
    this.toast(`Recording ${Math.round(total / 1000)}s at ${speed}×…`);
    setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, total);
  }

  private chromeTimer = 0;

  /** Show the transport, and schedule it to withdraw again. */
  private wakeChrome(holdMs = 2600) {
    if (document.body.dataset.chrome !== 'active') document.body.dataset.chrome = 'active';
    clearTimeout(this.chromeTimer);
    this.chromeTimer = window.setTimeout(() => {
      // Never hide it while the answer panel is up: at that point the buttons are the
      // whole point of the screen.
      if (this.mode === 'result' || this.scrubbing) {
        this.wakeChrome();
        return;
      }
      document.body.dataset.chrome = 'idle';
    }, holdMs);
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

  /**
   * Survive a lost GPU context.
   *
   * Backgrounding a tab, a driver reset, or memory pressure on a phone all take the
   * WebGL context away, and without a handler the canvas simply goes black forever with
   * no error and no way back. It is not an exotic case on mobile.
   *
   * The recovery is a reload rather than a rebuild. Every GL object this app holds
   * (programs, texture arrays, framebuffers, instance buffers) is invalid after a loss,
   * and re-creating them piecemeal in the right order is exactly the kind of code that
   * is written once, never exercised, and wrong when it finally runs. A reload is one
   * line and is correct by construction; the cost is the drawing, which is a fair trade
   * against a silently dead canvas.
   */
  private wireContextLoss() {
    const canvas = $<HTMLCanvasElement>('stage');

    canvas.addEventListener('webglcontextlost', (e) => {
      // Without preventDefault the browser will not attempt to restore the context.
      e.preventDefault();
      this.contextLost = true;
      this.audio.setEnabled(false);
      $('unsupportedTitle').textContent = 'The graphics context was lost';
      $('unsupported').querySelector('p')!.textContent =
        'This usually happens when the tab was in the background or the device ran low on memory. Reloading will start it again.';
      $('unsupported').hidden = false;
      $('app').hidden = true;
    });

    canvas.addEventListener('webglcontextrestored', () => {
      // The GPU is back, so a reload will come up immediately rather than failing again.
      window.location.reload();
    });
  }

  private loop = (now: number) => {
    requestAnimationFrame(this.loop);
    // Every buffer and program is invalid until the page reloads, so drawing would only
    // produce a stream of GL errors.
    if (this.contextLost) return;
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

    // In the same task as the GL draw. The context has no preserved drawing buffer, so
    // copying it after the browser composites would read back transparent black.
    if (this.recording) this.paintRecordedFrame();
  };

  /**
   * Compose one frame for the recorder: the scene, then the words over it.
   *
   * The text is read from the DOM the app has already filled in rather than rebuilt from
   * the run, so the clip cannot disagree with what the viewer is looking at.
   */
  private paintRecordedFrame() {
    const stage = $<HTMLCanvasElement>('stage');
    this.compositor.sync(stage, window.innerWidth);

    const box = (id: string) => {
      const b = $(id).getBoundingClientRect();
      return { left: b.left, top: b.top, width: b.width };
    };

    const caption =
      this.mode === 'reveal'
        ? {
            step: $('storyStep').textContent ?? '',
            title: $('storyTitle').textContent ?? '',
            caption: $('storyCaption').textContent ?? '',
            rect: box('story'),
          }
        : null;

    const fool = $('resultFool');
    const answer =
      this.mode === 'result'
        ? {
            digit: $('resultDigit').textContent ?? '',
            confidence: $('resultPct').textContent ?? '',
            phrase: $('resultPhrase').textContent ?? '',
            reason: $('resultReason').textContent ?? '',
            counter: $('resultCounter').textContent ?? '',
            fool: fool.hidden ? '' : (fool.textContent ?? ''),
            rect: box('result'),
          }
        : null;

    this.compositor.paint(stage, this.scene.labels, caption, answer, this.scene.stacked);
  }

  private lastStageKey = '';
  private lastCaption = '';
  private updateChrome(t: number) {
    if (!this.score) return;
    const stage = stageAt(this.score.stages, t);
    if (stage.key !== this.lastStageKey) {
      this.lastStageKey = stage.key;
      $('storyStep').textContent = `${stage.index + 1} / ${this.score.stages.length}`;
      $('storyTitle').textContent = stage.title;
    }

    // One block of text, in the place the reader already looks.
    //
    // Beat-specific claims temporarily replace the stage caption rather than appearing
    // in a second band of their own. Two simultaneous text blocks in different registers
    // split attention and read as clutter, however good each one is on its own.
    // `>=` so that when two claims overlap at full opacity the later one wins. NOTES is
    // declared in running order, and with a strict `>` the earlier claim would stick and
    // the newer beat would never get its line.
    let caption = stage.caption;
    let strongest = 0.45;
    for (const key in NOTES) {
      const weight = this.sampled[key] ?? 0;
      if (weight >= strongest) {
        strongest = weight;
        caption = NOTES[key];
      }
    }
    if (caption !== this.lastCaption) {
      this.lastCaption = caption;
      $('storyCaption').textContent = caption;
    }

    const pct = this.player.progress * 100;
    $('scrubFill').style.width = `${pct}%`;
    $('railFill').style.width = `${pct}%`;
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
