// node debug_fabric.cjs out.jpg [rounds...]: render relaxed canonical fabric face-on
const { chromium } = require('playwright'); const http = require('http'); const fs = require('fs'); const path = require('path');
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html' };
const srv = http.createServer((q, s) => { try { const f = path.join(__dirname, decodeURIComponent(q.url.split('?')[0])); s.writeHead(200, { 'Content-Type': mime[path.extname(f)] || 'application/octet-stream' }); s.end(fs.readFileSync(f)); } catch (e) { s.writeHead(404); s.end(); } }).listen(8795);
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage(); p.on('console', (m) => console.log('[page]', m.text())); p.on('pageerror', (e) => console.log('[err]', e.message));
  await p.goto('http://localhost:8795/web/index.html?scale=1'); await p.waitForFunction('window.ready');
  const views = JSON.parse(process.argv[3] || '[[0,0,110],[0,60,90]]');
  const imgs = [];
  for (const v of views) {
    const d = await p.evaluate(async (v) => {
      const { Sim } = await import('/engine/sim.mjs'); const { RING_CLOSED } = await import('/engine/fabric.mjs');
      const sim = new Sim({ rounds: ['mr 6', 'inc 6 to 18', 'even 1', 'dec 6 to 6'], actions: [{ type: 'hold', dur: 1 }], captions: [], cams: [{ ai: 0, shot: 'close' }] });
      sim.fab.ring = RING_CLOSED;
      const nSt = sim.fab.stitches.length; const upTo = +v[3] || (2 * nSt + 1);
      const pcs = sim.canonicalPieces(upTo);
      const pts = []; pcs.forEach((pc) => pts.push(...pc));
      const s = []; let acc = 0; pts.forEach((q, i) => { if (i) acc += Math.hypot(q[0] - pts[i - 1][0], q[1] - pts[i - 1][1], q[2] - pts[i - 1][2]); s.push(acc); });
      const { resampleAttr, laplace, limitCurvature } = await import('/engine/curve.mjs');
      let [P, S] = resampleAttr(pts, s, 0.32); P = laplace(P, 2, 0.35); 
      const { tubeRadii } = await import('/engine/sim.mjs'); return window.renderStatic({ pts: P, s: S, r: v[6] ? null : tubeRadii(P, P.map(() => 1)), three: !!v[5] }, { pos: [v[0], v[1], v[2]], look: [0, 0, -3] });
    }, v);
    imgs.push(d);
  }
  imgs.forEach((d, k) => fs.writeFileSync(process.argv[2].replace('.jpg', `_${k}.jpg`), Buffer.from(d.split(',')[1], 'base64')));
  await b.close(); srv.close();
})();
