import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Dev-only frame dump.
 *
 * The visualisation is a moving 3D scene, so the only honest way to review framing,
 * legibility and artefacts is to look at real frames. This lets the page POST a rendered
 * canvas straight to disk, which turns "does stage 4 read clearly?" into something
 * inspectable rather than something argued about.
 */
function frameDump(): Plugin {
  const outDir = path.resolve(process.cwd(), '.frames');
  return {
    name: 'frame-dump',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__frame', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('POST only');
          return;
        }
        // Cap the body. This only ever runs on a dev server, but an unbounded accumulator
        // behind an open endpoint is not something to leave in a public repository.
        const LIMIT = 32 * 1024 * 1024;
        const chunks: Buffer[] = [];
        let size = 0;
        req.on('data', (c: Buffer) => {
          size += c.length;
          if (size > LIMIT) {
            res.statusCode = 413;
            res.end('too large');
            req.destroy();
            return;
          }
          chunks.push(c);
        });
        req.on('end', () => {
          if (size > LIMIT) return;
          try {
            const { name, dataUrl } = JSON.parse(Buffer.concat(chunks).toString());
            const base64 = String(dataUrl).split(',')[1] ?? '';
            fs.mkdirSync(outDir, { recursive: true });
            const safe = String(name).replace(/[^a-z0-9._-]/gi, '_');
            fs.writeFileSync(path.join(outDir, safe), Buffer.from(base64, 'base64'));
            res.end(JSON.stringify({ ok: true, path: path.join(outDir, safe) }));
          } catch (err) {
            res.statusCode = 400;
            res.end(String(err));
          }
        });
      });
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [frameDump()],
  build: {
    target: 'es2022',
    assetsInlineLimit: 0, // keep model.bin and the wasm as separate cacheable files
    rollupOptions: {
      output: {
        // One chunk. The whole app is smaller than a typical framework runtime, and a
        // single request beats waterfalling for time-to-first-frame.
        manualChunks: undefined,
      },
    },
  },
  server: { port: 5173, host: true },
  preview: { port: 4173, host: true },
});
