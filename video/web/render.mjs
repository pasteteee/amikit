// Browser renderer: steps the simulation and draws one 1080x1920 frame with captions.
import * as THREE from 'three';
import { RoomEnvironment } from '/node_modules/three/examples/jsm/environments/RoomEnvironment.js';
import { Sim } from '/engine/sim.mjs';
import { YarnFollower } from '/engine/follow.mjs';
let follower = null;
import { surfaceRadius, HOOK } from '/engine/hook.mjs';
import { RY } from '/engine/fabric.mjs';
import { cameraAt, captionAt, badgeAt } from '/engine/director.mjs';
import { shortCamera, shortOverlay } from '/engine/short.mjs';
import { drawShortOverlay, loadShortAssets } from '/web/overlay2.mjs';
import { fillTube } from '/engine/tube.mjs';

const W = 1080, H = 1920;
const params = new URLSearchParams(location.search);
const scale = +(params.get('scale') || 1);

const gl = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
gl.setPixelRatio(1);
gl.setSize(W * scale, H * scale);
gl.shadowMap.enabled = true;
gl.shadowMap.type = THREE.PCFSoftShadowMap;
gl.toneMapping = THREE.ACESFilmicToneMapping;
gl.toneMappingExposure = 0.92;
gl.outputColorSpace = THREE.SRGBColorSpace;

const out = document.createElement('canvas');
out.width = W * scale; out.height = H * scale;
document.body.appendChild(out);
const ctx = out.getContext('2d');

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(gl);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

// soft studio backdrop (gradient sphere)
const skyMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide, depthWrite: false });
function setBackdrop(stops) {
  const c = document.createElement('canvas'); c.width = 16; c.height = 512;
  const g = c.getContext('2d');
  const gr = g.createLinearGradient(0, 0, 0, 512);
  stops.forEach(([p, col]) => gr.addColorStop(p, col));
  g.fillStyle = gr; g.fillRect(0, 0, 16, 512);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  skyMat.map = tex; skyMat.needsUpdate = true;
}
setBackdrop([[0, '#f6efe6'], [0.55, '#ead9c6'], [1, '#cdb49b']]);
scene.add(new THREE.Mesh(new THREE.SphereGeometry(900, 32, 16), skyMat));

const camera = new THREE.PerspectiveCamera(28, W / H, 2, 3000);

const key = new THREE.DirectionalLight(0xfff3e6, 2.4);
key.position.set(-40, 70, 90);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
const sc = key.shadow.camera; sc.left = -45; sc.right = 45; sc.top = 45; sc.bottom = -45; sc.near = 10; sc.far = 300;
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.03; key.shadow.radius = 3;
scene.add(key); scene.add(key.target);
const rim = new THREE.DirectionalLight(0xdfe8ff, 0.9); rim.position.set(60, 30, -60); scene.add(rim);
scene.add(new THREE.HemisphereLight(0xfff8ef, 0x8a6f58, 0.55));

// ---------- yarn material: twisted plies via procedural bump + colour texture
function plyTextures() {
  const w = 256, h = 128;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d');
  const img = g.createImageData(w, h);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const fib = new Float32Array(w * h);
  for (let k = 0; k < 900; k++) { // fibre streaks along the ply direction
    let x = rnd() * w, y = rnd() * h; const L = 20 + rnd() * 60, a = rnd() * 0.4 - 0.2;
    for (let s = 0; s < L; s++) { const xi = ((Math.round(x + s) % w) + w) % w, yi = ((Math.round(y + s * (h / w) * 2 * 1 + a * s) % h) + h) % h; fib[yi * w + xi] += 0.25; }
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    // 3 plies, one twist per texture repeat along x; v (around) is y
    const f = ((y / h) * 3 + (x / w) * 3) % 1;
    const ply = Math.sin(Math.PI * f); // 0 at grooves, 1 at ply centre
    const v = Math.pow(ply, 0.6) * 0.85 + Math.min(1, fib[y * w + x]) * 0.15;
    const i = (y * w + x) * 4;
    const b = Math.round(255 * v);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = b; img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const bump = new THREE.CanvasTexture(c);
  bump.wrapS = bump.wrapT = THREE.RepeatWrapping;
  bump.anisotropy = 8;
  // colour/occlusion map: darker in the grooves
  const c2 = document.createElement('canvas'); c2.width = w; c2.height = h;
  const g2 = c2.getContext('2d'); const im2 = g2.createImageData(w, h);
  for (let k = 0; k < w * h; k++) { const v = img.data[k * 4] / 255; const o = 0.86 + 0.14 * v; im2.data[k * 4] = im2.data[k * 4 + 1] = im2.data[k * 4 + 2] = Math.round(255 * o); im2.data[k * 4 + 3] = 255; }
  g2.putImageData(im2, 0, 0);
  const ao = new THREE.CanvasTexture(c2); ao.wrapS = ao.wrapT = THREE.RepeatWrapping; ao.colorSpace = THREE.SRGBColorSpace; ao.anisotropy = 8;
  return { bump, ao };
}
const ply = plyTextures();
const PLY_PITCH = 2.6; // mm per full twist of the plies

let yarnColor = new THREE.Color('#ddbe98');
const yarnMat = new THREE.MeshPhysicalMaterial({
  color: yarnColor, map: ply.ao, bumpMap: ply.bump, bumpScale: 0.45, roughness: 0.88, metalness: 0,
  sheen: 0.6, sheenRoughness: 0.5, sheenColor: new THREE.Color('#e9d6bd'),
});

// dynamic tube geometry
const RADIAL = 10;
let yarnGeo = new THREE.BufferGeometry();
let yarnCap = 0;
const yarnMesh = new THREE.Mesh(yarnGeo, yarnMat);
yarnMesh.castShadow = true; yarnMesh.receiveShadow = true;
yarnMesh.frustumCulled = false;
scene.add(yarnMesh);

function buildTube(pts, s, rad) {
  const n = pts.length;
  if (n > yarnCap || n < yarnCap * 0.5 || true) {
    yarnCap = n;
    yarnGeo.dispose();
    yarnGeo = new THREE.BufferGeometry();
    yarnGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(yarnCap * (RADIAL + 1) * 3), 3));
    yarnGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(yarnCap * (RADIAL + 1) * 3), 3));
    yarnGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(yarnCap * (RADIAL + 1) * 2), 2));
    const idx = new Uint32Array((yarnCap - 1) * RADIAL * 6);
    yarnGeo.setIndex(new THREE.BufferAttribute(idx, 1));
    yarnMesh.geometry = yarnGeo;
  }
  const P = yarnGeo.attributes.position.array, N = yarnGeo.attributes.normal.array, U = yarnGeo.attributes.uv.array, I = yarnGeo.index.array;
  const q = fillTube(pts, s, rad, RADIAL, RY, PLY_PITCH, P, N, U, I);
  yarnGeo.setDrawRange(0, q);
  yarnGeo.attributes.position.needsUpdate = true; yarnGeo.attributes.normal.needsUpdate = true; yarnGeo.attributes.uv.needsUpdate = true; yarnGeo.index.needsUpdate = true;
  yarnGeo.computeBoundingSphere();
}
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

const hookMetal = new THREE.MeshPhysicalMaterial({ color: '#c7a2d6', metalness: 0.85, roughness: 0.28, clearcoat: 0.6, clearcoatRoughness: 0.2 });
const hookHandle = new THREE.MeshPhysicalMaterial({ color: '#4f8f86', roughness: 0.6, metalness: 0, sheen: 0.3 });
// ---------- hook mesh (hook-local: +x toward tip, +y throat side)
function hookMesh() {
  const group = new THREE.Group();
  const make = (x0, x1, nx, mat) => {
    const PHI = 40;
    const pos = [], uv = [], idx = [];
    const xs = [];
    for (let i = 0; i <= nx; i++) { const t = i / nx; xs.push(x0 + (x1 - x0) * (1 - Math.pow(1 - t, 2.2))); }
    xs.forEach((x, i) => {
      for (let k = 0; k <= PHI; k++) {
        const phi = (k / PHI) * Math.PI * 2;
        let r = surfaceRadius(x, phi);
        let sy = 1, sz = 1;
        if (x < HOOK.thumbStart && x > HOOK.thumbEnd) { const e = Math.sin(Math.PI * (x - HOOK.thumbStart) / (HOOK.thumbEnd - HOOK.thumbStart)); sy = 1 - 0.45 * e; sz = 1 + 1.6 * e; }
        pos.push(x, r * Math.cos(phi) * sy, r * Math.sin(phi) * sz);
        uv.push(i / nx, k / PHI);
      }
    });
    for (let i = 0; i < xs.length - 1; i++) for (let k = 0; k < PHI; k++) {
      const a = i * (PHI + 1) + k, b = a + PHI + 1;
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx); g.computeVertexNormals();
    const m = new THREE.Mesh(g, mat); m.castShadow = true; m.receiveShadow = true;
    group.add(m);
  };
  const metal = hookMetal, handle = hookHandle;
  make(HOOK.handleStart + 0.5, 0.0, 420, metal);
  make(-200, HOOK.handleStart + 0.6, 30, handle);
  return group;
}
const hook = hookMesh();
scene.add(hook);
function poseHook(hk) {
  const a = hk.a, d = hk.d; const s = cross3(a, d);
  const m = new THREE.Matrix4().set(a[0], d[0], s[0], hk.tip[0], a[1], d[1], s[1], hk.tip[1], a[2], d[2], s[2], hk.tip[2], 0, 0, 0, 1);
  hook.matrixAutoUpdate = false; hook.matrix.copy(m); hook.matrixWorldNeedsUpdate = true;
}

// ---------- captions
function wrapText(text, maxW) {
  const words = text.split(' '); const lines = []; let cur = '';
  for (const w of words) { const t = cur ? cur + ' ' + w : w; if (ctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t; }
  if (cur) lines.push(cur); return lines;
}
function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
function drawOverlay(state) {
  ctx.save(); ctx.scale(scale, scale);
  const { cap, badge, title } = state;
  if (title) {
    ctx.globalAlpha = title.alpha;
    ctx.font = '800 64px Inter, sans-serif'; ctx.fillStyle = '#3b2a20'; ctx.textAlign = 'center';
    const lines = wrapText(title.text, 940);
    lines.forEach((l, k) => ctx.fillText(l, W / 2, 190 + k * 76));
    if (title.sub) { ctx.font = '500 38px Inter, sans-serif'; ctx.fillStyle = '#6b5446'; ctx.fillText(title.sub, W / 2, 200 + lines.length * 76 + 6); }
    ctx.globalAlpha = 1;
  }
  if (cap && cap.alpha > 0.01) {
    ctx.globalAlpha = cap.alpha;
    ctx.font = '600 50px Inter, sans-serif';
    const lines = wrapText(cap.text, 900);
    const boxH = 70 + lines.length * 64 + (cap.head ? 70 : 0);
    const y0 = H - 230 - boxH;
    roundRect(60, y0, W - 120, boxH, 36); ctx.fillStyle = 'rgba(255,250,244,0.9)'; ctx.fill();
    let y = y0 + 40;
    if (cap.head) { ctx.font = '800 44px Inter, sans-serif'; ctx.fillStyle = '#b4583c'; ctx.textAlign = 'left'; ctx.fillText(cap.head, 100, y + 38); y += 70; }
    ctx.font = '600 50px Inter, sans-serif'; ctx.fillStyle = '#2d221b'; ctx.textAlign = 'left';
    lines.forEach((l, k) => ctx.fillText(l, 100, y + 46 + k * 64));
    ctx.globalAlpha = 1;
  }
  if (badge) {
    ctx.font = '800 46px Inter, sans-serif'; ctx.textAlign = 'center';
    const tw = ctx.measureText(badge).width + 56;
    roundRect(W - 70 - tw, 90, tw, 78, 39); ctx.fillStyle = 'rgba(45,34,27,0.82)'; ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText(badge, W - 70 - tw / 2, 145);
  }
  ctx.restore();
}

// ---------- driver
let sim = null, scn = null, frame = -1, opts2 = {};
window.init = async (scenario, opts = {}) => {
  scn = scenario; opts2 = { ...opts, meta: scenario.meta };
  if (opts.colour) yarnMat.color = new THREE.Color(opts.colour);
  if (opts.sheen) yarnMat.sheenColor = new THREE.Color(opts.sheen);
  if (opts.hookColour) hookMetal.color = new THREE.Color(opts.hookColour);
  if (opts.handleColour) hookHandle.color = new THREE.Color(opts.handleColour);
  if (opts.backdrop) setBackdrop(opts.backdrop);
  if (opts.exposure) gl.toneMappingExposure = opts.exposure;
  if (scenario.style === 'short2') await loadShortAssets(opts);
  sim = new Sim(scn);
  follower = scn.meta?.follow ? new YarnFollower(scn.meta.follow === true ? {} : scn.meta.follow) : null;
  frame = -1;
  return { duration: sim.duration, frames: Math.ceil(sim.duration * sim.fps) };
};
window.renderFrame = (f, draw = true) => {
  let st;
  while (frame < f) { frame++; st = sim.step(frame / sim.fps); if (follower) st = follower.step(st, 1 / sim.fps); }
  if (!draw) return null;
  buildTube(st.pts, st.s, st.r);
  poseHook(st.hook);
  const short = scn.style === 'short2';
  const c = short ? shortCamera(sim, frame / sim.fps, st) : cameraAt(sim, frame / sim.fps);
  camera.position.set(...c.pos); camera.fov = c.fov; camera.updateProjectionMatrix(); camera.lookAt(...c.look);
  key.target.position.set(...c.look);
  key.position.set(c.look[0] - 40, c.look[1] + 70, c.look[2] + 90);
  gl.render(scene, camera);
  ctx.clearRect(0, 0, out.width, out.height);
  ctx.drawImage(gl.domElement, 0, 0);
  if (short) {
    const ov = shortOverlay(sim, frame / sim.fps, st);
    const proj = (p) => { const v = new THREE.Vector3(...p).project(camera); return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H]; };
    const h = st.hook;
    ov.throat = proj([h.tip[0] + h.a[0] * HOOK.xThroat, h.tip[1] + h.a[1] * HOOK.xThroat, h.tip[2] + h.a[2] * HOOK.xThroat]);
    ov.pullTo = proj([h.tip[0] - h.a[0] * 9, h.tip[1] - h.a[1] * 9, h.tip[2] - h.a[2] * 9]);
    ov.t = frame / sim.fps;
    drawShortOverlay(ctx, scale, ov, opts2);
  } else drawOverlay({ cap: captionAt(sim, frame / sim.fps), badge: badgeAt(sim, frame / sim.fps), title: null });
  return out.toDataURL('image/jpeg', 0.92);
};
window.setLook = (opts) => {
  if (opts.colour) yarnMat.color = new THREE.Color(opts.colour);
  if (opts.sheen) yarnMat.sheenColor = new THREE.Color(opts.sheen);
  if (opts.hookColour) hookMetal.color = new THREE.Color(opts.hookColour);
  if (opts.handleColour) hookHandle.color = new THREE.Color(opts.handleColour);
  if (opts.backdrop) setBackdrop(opts.backdrop);
};
window.renderStatic = (st, c) => {
  if (st.three) {
    const curve = new THREE.CatmullRomCurve3(st.pts.map((p) => new THREE.Vector3(...p)));
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, st.pts.length * 2, 0.7, 10), yarnMat); scene.add(m); yarnMesh.visible = false;
  } else buildTube(st.pts, st.s, st.r);
  if (st.hook) { hook.visible = true; poseHook(st.hook); } else hook.visible = false;
  camera.position.set(...c.pos); camera.fov = c.fov || 28; camera.updateProjectionMatrix(); camera.lookAt(...c.look);
  key.target.position.set(...c.look); key.position.set(c.look[0] - 40, c.look[1] + 70, c.look[2] + 90);
  gl.render(scene, camera);
  ctx.clearRect(0, 0, out.width, out.height); ctx.drawImage(gl.domElement, 0, 0);
  return out.toDataURL('image/jpeg', 0.92);
};
window.ready = true;
