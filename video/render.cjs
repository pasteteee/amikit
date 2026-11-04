// Usage: node render.cjs <scenario.json> <outdir> [from] [to] [step] [scale]
const { chromium } = require('playwright'); const http = require('http'); const fs = require('fs'); const path = require('path');
const [, , scnPath, outDir, from = '0', to = '-1', step = '1', scale = '1', port = '8790'] = process.argv;
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json' };
const srv = http.createServer((q, s) => { try { const f = path.join(__dirname, decodeURIComponent(q.url.split('?')[0])); s.writeHead(200, { 'Content-Type': mime[path.extname(f)] || 'application/octet-stream' }); s.end(fs.readFileSync(f)); } catch (e) { s.writeHead(404); s.end(); } }).listen(+port);
(async () => {
  const scn = JSON.parse(fs.readFileSync(scnPath, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 540, height: 960 } });
  p.on('console', (m) => console.log('[page]', m.text())); p.on('pageerror', (e) => console.log('[err]', e.message));
  await p.goto(`http://localhost:${port}/web/index.html?scale=${scale}`); await p.waitForFunction('window.ready');
  const info = await p.evaluate(([s, o]) => window.init(s, o), [scn, scn.meta?.render || {}]);
  const end = +to < 0 ? info.frames - 1 : Math.min(+to, info.frames - 1);
  console.log('frames', info.frames, 'duration', info.duration.toFixed(1));
  const t0 = Date.now(); let n = 0;
  for (let f = +from; f <= end; f += +step) {
    const d = await p.evaluate((f) => window.renderFrame(f), f);
    fs.writeFileSync(path.join(outDir, `f${String(f).padStart(5, '0')}.jpg`), Buffer.from(d.split(',')[1], 'base64')); n++;
    if (n % 25 === 0) console.log('frame', f, ((Date.now() - t0) / n).toFixed(0), 'ms/frame');
  }
  await b.close(); srv.close();
})();
