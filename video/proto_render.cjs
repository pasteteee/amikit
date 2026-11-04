// Usage: node proto_render.cjs <frames.json> <outdir> [step] [scale] [port] -- renders prototype frames
const { chromium } = require('playwright'); const http = require('http'); const fs = require('fs'); const path = require('path');
const [, , inPath, outDir, step = '1', scale = '0.5', port = '8797'] = process.argv;
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.woff2': 'font/woff2' };
const srv = http.createServer((q, s) => { try { const f = path.join(__dirname, decodeURIComponent(q.url.split('?')[0])); s.writeHead(200, { 'Content-Type': mime[path.extname(f)] || 'application/octet-stream' }); s.end(fs.readFileSync(f)); } catch (e) { s.writeHead(404); s.end(); } }).listen(+port);
(async () => {
  const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 540, height: 960 } });
  p.on('pageerror', (e) => console.log('[err]', e.message));
  await p.goto(`http://localhost:${port}/web/index.html?scale=${scale}`); await p.waitForFunction('window.ready');
  await p.evaluate(() => window.setLook({ colour: '#f77ea0', sheen: '#ffd0dc', hookColour: '#7fd6c8', handleColour: '#4b2a5e', backdrop: [[0, '#fbf3ea'], [0.5, '#f4e4d4'], [1, '#e6cbb4']] }));
  const cam = data.cam || { pos: [40, 30, 70], look: [3, 4, 0], fov: 30 };
  for (let f = 0; f < data.frames.length; f += +step) {
    const fr = data.frames[f];
    const st = { pts: fr.pts, s: fr.s, r: fr.pts.map(() => data.r), hook: fr.hook };
    const d = await p.evaluate(([st, c]) => window.renderStatic(st, c), [st, cam]);
    fs.writeFileSync(path.join(outDir, `f${String(f).padStart(5, '0')}.jpg`), Buffer.from(d.split(',')[1], 'base64'));
  }
  await b.close(); srv.close();
})();
