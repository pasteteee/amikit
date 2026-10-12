// amikit editor: three.js viewer + spec panel. Every change is a list of edit ops (amikit/edit.py) sent to the
// server, which validates and builds the toy; the GLB comes back and replaces the scene.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (s, el = document) => el.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (v !== undefined && v !== null && v !== false) el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined) el.append(c.nodeType ? c : String(c));
  return el;
};

// ---------------------------------------------------------------------------------------------- state
const S = {
  spec: null, summary: "", info: { parts: {} }, toyId: null,
  history: [], hi: -1,
  tab: "parts", selPart: null, selEntry: null, selFace: null,
  pending: new Map(), timer: null, inflight: false,
  lastBuild: null, cmd: null,
};

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data;
  try { data = await r.json(); } catch { data = { error: `HTTP ${r.status}` }; }
  if (!r.ok) { const e = new Error(data.error || `HTTP ${r.status}`); e.status = r.status; throw e; }
  return data;
}

function parseSummary(txt) {
  const parts = {};
  for (const ln of (txt || "").split("\n")) {
    const m = ln.match(/^part (\S+): (\d+) rounds, max (\d+) st/);
    if (m) parts[m[1]] = { rounds: +m[2], max: +m[3] };
  }
  return { parts };
}

function showErr(msg) {
  const e = $("#err");
  if (!msg) { e.classList.add("hidden"); return; }
  e.textContent = msg; e.classList.remove("hidden");
}
function setBusy(on, text = "сборка…") { const b = $("#busy"); b.textContent = text; b.classList.toggle("hidden", !on); }

// --------------------------------------------------------------------------------------------- history
function pushHistory(spec) {
  S.history = S.history.slice(0, S.hi + 1);
  S.history.push(JSON.stringify(spec));
  if (S.history.length > 200) S.history.shift();
  S.hi = S.history.length - 1;
  updUndo();
}
function updUndo() { $("#undoBtn").disabled = S.hi <= 0; $("#redoBtn").disabled = S.hi >= S.history.length - 1; }
async function stepHistory(d) {
  const i = S.hi + d;
  if (i < 0 || i >= S.history.length) return;
  S.pending.clear(); clearTimeout(S.timer);
  S.hi = i; updUndo();
  S.spec = JSON.parse(S.history[i]);
  renderPanel();
  await rebuild();
}

// ------------------------------------------------------------------------------------- edits and builds
// queue a 'set' (merged by path) and send after 400 ms of quiet
function setOp(path, value) { queueOps([{ op: "set", path, value }]); }
function queueOps(ops, now = false) {
  for (const op of ops) S.pending.set(op.op === "set" ? "set:" + op.path : "x:" + Math.random(), op);
  clearTimeout(S.timer);
  S.timer = setTimeout(flush, now ? 0 : 400);
}

async function flush() {
  if (S.inflight) { S.timer = setTimeout(flush, 150); return; }
  if (!S.pending.size) return;
  const ops = [...S.pending.values()]; S.pending.clear();
  S.inflight = true; setBusy(true);
  try {
    const res = await api("/api/edit", { spec: S.spec, ops, build: true });
    S.spec = res.spec; S.summary = res.summary; S.info = parseSummary(res.summary);
    pushHistory(S.spec); showErr(null);
    await showBuild(res.build);
    // re-render the panel unless the user is dragging a slider / colour picker (keeps the drag alive)
    const ae = document.activeElement;
    if (ae && ae.tagName === "INPUT" && (ae.type === "range" || ae.type === "color") && ops.every(o => o.op === "set")) refreshInfo();
    else renderPanel();
  } catch (e) {
    showErr("Правка не применена: " + e.message);
    renderPanel(); // revert the controls to the current spec
  } finally { S.inflight = false; setBusy(false); }
}

async function commitSpec(spec, { history = true } = {}) {
  // a whole new spec (JSON tab, LLM command, load): validate via edit with no ops, then build
  S.pending.clear(); clearTimeout(S.timer);
  setBusy(true);
  try {
    const res = await api("/api/edit", { spec, ops: [], build: true });
    S.spec = res.spec; S.summary = res.summary; S.info = parseSummary(res.summary);
    if (history) pushHistory(S.spec);
    showErr(null); renderPanel();
    await showBuild(res.build);
    return true;
  } catch (e) { showErr(e.message); return false; }
  finally { setBusy(false); }
}

async function rebuild() {
  setBusy(true);
  try {
    const b = await api("/api/build", { spec: S.spec });
    S.summary = b.summary; S.info = parseSummary(b.summary);
    showErr(null); await showBuild(b); refreshInfo();
  } catch (e) { showErr(e.message); }
  finally { setBusy(false); }
}

async function showBuild(b) {
  S.lastBuild = b;
  const t = b.timings || {};
  $("#status").textContent = `сборка ${t.cached ? "из кэша" : (t.total ?? "?") + " с"} · ${Math.round((b.tris || 0) / 1000)}k треуг. · флагов: ${b.flags.length}`;
  $("#flags").replaceChildren(...b.flags.map(f => h("div", {}, f)));
  $("#patternBtn").disabled = !b.pattern_url;
  $("#patternBtn").title = b.pattern_url ? "Текстовая схема вязания" : "Схема недоступна (нет amikit/pattern_text.py)";
  await loadGLB(b.glb_url);
}

// -------------------------------------------------------------------------------------------- 3D view
const canvas = $("#c3d");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.NoToneMapping;
const scene = new THREE.Scene();
scene.background = new THREE.Color("#d9d2cc");
const camera = new THREE.PerspectiveCamera(30, 1, 1, 5000);
camera.position.set(120, 140, 420);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xfff6ee, 0x9a8f88, 2.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-160, 300, 220);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.radius = 6; sun.shadow.bias = -0.0005;
scene.add(sun, sun.target);
const fill = new THREE.DirectionalLight(0xfff0e6, 0.5); fill.position.set(250, 80, 150); scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.ShadowMaterial({ opacity: 0.22 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

let toyRoot = null, firstFit = true;
const loader = new GLTFLoader();
const toLin = c => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

function loadGLB(url) {
  return new Promise((resolve, reject) => loader.load(url + "?t=" + Date.now(), gltf => {
    if (toyRoot) { scene.remove(toyRoot); toyRoot.traverse(o => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } }); }
    toyRoot = gltf.scene;
    toyRoot.traverse(o => {
      if (!o.isMesh) return;
      // GLB vertex colours are display (sRGB) values; three.js expects linear
      const col = o.geometry.getAttribute("color");
      if (col) {
        const n = col.count, arr = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) { arr[3 * i] = toLin(col.getX(i)); arr[3 * i + 1] = toLin(col.getY(i)); arr[3 * i + 2] = toLin(col.getZ(i)); }
        o.geometry.setAttribute("color", new THREE.BufferAttribute(arr, 3));
      }
      const name = meshName(o);
      const shiny = name.includes("@") && /^(eye|nose)/.test(name);
      o.material = new THREE.MeshStandardMaterial({ vertexColors: !!col, color: col ? 0xffffff : 0x333333,
        roughness: shiny ? 0.25 : 0.92, metalness: 0 });
      o.castShadow = true; o.receiveShadow = true;
      o.userData.inst = name;
    });
    scene.add(toyRoot);
    const box = new THREE.Box3().setFromObject(toyRoot);
    const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
    const R = Math.max(sz.x, sz.y, sz.z);
    Object.assign(sun.shadow.camera, { left: -R, right: R, top: R, bottom: -R, near: 1, far: 2000 });
    sun.shadow.camera.updateProjectionMatrix();
    sun.target.position.copy(c);
    if (firstFit) {
      firstFit = false;
      controls.target.copy(c);
      // fit the bounding sphere into both the vertical and the horizontal field of view
      const rad = sz.length() / 2, vf = THREE.MathUtils.degToRad(camera.fov) / 2;
      const hf = Math.atan(Math.tan(vf) * camera.aspect);
      const dist = rad / Math.sin(Math.min(vf, hf)) * 1.05;
      camera.position.copy(c).add(new THREE.Vector3(0.35, 0.25, 1).normalize().multiplyScalar(dist));
      controls.update();
    }
    highlight();
    resolve();
  }, undefined, reject));
}

function meshName(o) {
  // node names: instance names (ear_L) or face meshes ("eye@head"); GLTFLoader may add _1, _2 for duplicates
  let n = o.name || (o.parent && o.parent.name) || "";
  return n;
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// picking: click (not drag) on a mesh selects its part
const ray = new THREE.Raycaster();
let downAt = null;
canvas.addEventListener("pointerdown", e => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4 || !toyRoot) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObject(toyRoot, true)[0];
  if (!hit) { select(null); return; }
  selectInstance(hit.object.userData.inst);
});

function entryName(e) { return e.name || e.part; }
function instanceMap() {
  // instance name -> assembly entry
  const m = {};
  for (const e of S.spec.assembly || []) {
    const n = entryName(e);
    if (e.mirror) { m[e.name || e.part + "_L"] = e; m[e.name_r || e.part + "_R"] = e; } else m[n] = e;
  }
  return m;
}
function selectInstance(inst) {
  if (!inst) return;
  if (inst.includes("@")) {          // face item: "<kind>@<host instance>"
    const [kind, hostRaw] = inst.split("@");
    const hostRaw2 = hostRaw.replace(/#\d+$/, "").replace(/_\d+$/, "");
    const host = hostRaw2.replace(/_[LR]$/, "");
    const items = S.spec.face || [];
    const idx = items.findIndex(f => (f.on === host || f.on === hostRaw2) && kindOf(f) === kind);
    S.selFace = idx >= 0 ? faceKey(items[idx], idx) : null;
    $("#pick").textContent = `лицо: ${kind} на ${host}`;
    switchTab("face");
    return;
  }
  const im = instanceMap();
  let e = im[inst] || im[inst.replace(/_\d+$/, "")];
  const part = e ? e.part : inst.replace(/_\d+$/, "").replace(/_[LR]$/, "");
  select(part, e ? entryName(e) : null);
  $("#pick").textContent = `выбрано: ${inst}`;
}
function kindOf(f) {
  return { safety_eye: "eye", eye: "eye", safety_nose: "nose", stitch: "thread", embroidery: "thread", line: "thread",
           satin: "satin", blush: "blush", felt: "felt" }[f.type] || f.type;
}
function select(part, entry = null) {
  S.selPart = part; S.selEntry = entry ?? ((S.spec.assembly || []).find(e => e.part === part) ? entryName((S.spec.assembly || []).find(e => e.part === part)) : null);
  if (!part) $("#pick").textContent = "";
  if (S.tab !== "parts" && S.tab !== "asm") switchTab("parts"); else renderPanel();
  highlight();
  const card = document.querySelector(`.card.sel`);
  // scroll only the panel's own list (never the page: in the stacked layout that would hide the 3D view)
  const body = $("#tabBody");
  if (card && body.scrollHeight > body.clientHeight + 4) {
    const dy = card.getBoundingClientRect().top - body.getBoundingClientRect().top - 8;
    body.scrollTo({ top: body.scrollTop + dy, behavior: "smooth" });
  }
}
function highlight() {
  if (!toyRoot) return;
  const insts = new Set();
  if (S.selPart) for (const [n, e] of Object.entries(instanceMap())) if (e.part === S.selPart) insts.add(n);
  toyRoot.traverse(o => {
    if (!o.isMesh) return;
    const n = (o.userData.inst || "").replace(/_\d+$/, "");
    const on = insts.has(n) || insts.has(o.userData.inst);
    o.material.emissive = new THREE.Color(on ? 0x5a2a08 : 0x000000);
    o.material.emissiveIntensity = on ? 0.35 : 0;
  });
}

// ------------------------------------------------------------------------------------------- panel UI
function switchTab(t) {
  S.tab = t;
  for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("on", b.dataset.tab === t);
  renderPanel();
}
for (const b of document.querySelectorAll("#tabs button")) b.addEventListener("click", () => switchTab(b.dataset.tab));

const fmt = v => (Number.isInteger(+v) ? String(+v) : (+v).toFixed(Math.abs(v) >= 10 ? 1 : 2).replace(/0+$/, "").replace(/\.$/, ""));

// slider row: label, value, min, max, step, onchange(value)
function slider(label, value, min, max, step, onchange, { unset = false } = {}) {
  const val = h("span", { class: "val" }, unset ? "авто" : fmt(value));
  const inp = h("input", { type: "range", min, max, step, value: unset ? (min + max) / 2 : value });
  if (unset) inp.style.opacity = 0.45;
  inp.addEventListener("input", () => { val.textContent = fmt(inp.value); inp.style.opacity = 1; onchange(+inp.value); });
  return h("div", { class: "fld" }, h("label", {}, label), inp, val);
}
function selectFld(label, value, options, onchange) {
  const s = h("select", {}, options.map(([v, t]) => h("option", { value: v, selected: String(v) === String(value) }, t)));
  s.addEventListener("change", () => onchange(s.value));
  return h("div", { class: "fld wide" }, h("label", {}, label), s);
}
function allColours() {
  const c = { ...(S.spec.colours || {}) };
  if (!c.black) c.black = "#151313"; if (!c.white) c.white = "#f4f1ea";
  return c;
}
function colourOpts(withNone) {
  return [...(withNone ? [["", "—"]] : []), ...Object.keys(allColours()).map(k => [k, k])];
}

function renderPanel() {
  if (!S.spec) return;
  const body = $("#tabBody");
  const scroll = body.scrollTop;
  body.replaceChildren(({ parts: tabParts, asm: tabAsm, face: tabFace, colours: tabColours, json: tabJson })[S.tab]());
  body.scrollTop = scroll;
  updUndo();
}
function refreshInfo() {
  for (const el of document.querySelectorAll("[data-info]")) {
    const i = S.info.parts[el.dataset.info];
    if (i) el.textContent = `${i.rounds} р., макс ${i.max} п.`;
  }
}

function card(title, sel, onhead, content, extra = []) {
  const c = h("div", { class: "card" + (sel ? " sel open" : "") },
    h("div", { class: "hd", onclick: () => onhead() }, ...title, ...extra), h("div", { class: "bd" }, content));
  return c;
}

// (a) parts
function tabParts() {
  const cols = allColours();
  return h("div", {}, (S.spec.parts || []).map(p => {
    const sel = p.id === S.selPart;
    const pre = `parts.${p.id}.`;
    const info = S.info.parts[p.id];
    const ta = h("textarea", { class: "rounds", rows: Math.max(3, p.rounds.length + 1) }, p.rounds.join("\n"));
    const applyRounds = () => {
      const lines = ta.value.split("\n").map(s => s.trim()).filter(Boolean).map(s => (/^\d+$/.test(s) ? +s : s));
      if (JSON.stringify(lines) !== JSON.stringify(p.rounds)) queueOps([{ op: "set", path: pre + "rounds", value: lines }], true);
    };
    ta.addEventListener("change", applyRounds);
    ta.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); applyRounds(); } });
    const end = p.end || "closed";
    return card(
      [h("span", { class: "swatch", style: `background:${cols[p.colour] || "#ccc"}` }), h("b", {}, p.id)],
      sel, () => select(sel ? null : p.id),
      [
        h("div", { class: "fld wide" }, h("label", {}, "Ряды"), h("div", {}, ta, h("div", { class: "muted" }, "по строке на шаг; Ctrl+Enter или выход из поля — применить"))),
        selectFld("Цвет", p.colour, colourOpts(false), v => setOp(pre + "colour", v)),
        selectFld("Конец", end, [["closed", "закрыт (closed)"], ["open", "открыт (open)"]], v => setOp(pre + "end", v)),
        selectFld("Набивка", p.stuff || (end === "closed" ? "firm" : "none"), [["firm", "плотная"], ["medium", "средняя"], ["light", "лёгкая"], ["none", "нет"]],
          v => setOp(pre + "stuff", v)),
        slider("Сплющить", p.flatten || 0, 0, 1, 0.05, v => setOp(pre + "flatten", v)),
        slider("Защип", p.squash || 0, 0, 1, 0.05, v => setOp(pre + "squash", v)),
        h("div", { class: "row" },
          h("button", { onclick: () => queueOps([{ op: "scale_part", part: p.id, factor: 0.9 }], true) }, "− 10%"),
          h("button", { onclick: () => queueOps([{ op: "scale_part", part: p.id, factor: 1.1 }], true) }, "+ 10%"),
          h("span", { style: "flex:1" }),
          h("button", { class: "danger", onclick: () => { if (confirm(`Удалить деталь «${p.id}» и всё, что к ней пришито?`)) queueOps([{ op: "remove", path: "parts." + p.id }], true); } }, "Удалить")),
      ],
      [h("span", { class: "info", "data-info": p.id }, info ? `${info.rounds} р., макс ${info.max} п.` : "")]);
  }));
}

// (b) assembly
function partRounds(onName) {
  // `on` is an entry name (or instance); find its part's round count
  const e = (S.spec.assembly || []).find(x => entryName(x) === onName) || (S.spec.assembly || []).find(x => x.part === onName);
  const pid = e ? e.part : onName;
  return (S.info.parts[pid] || {}).rounds || 30;
}
function tabAsm() {
  return h("div", {}, (S.spec.assembly || []).map(e => {
    const n = entryName(e), pre = `assembly.${n}.`;
    const sel = e.part === S.selPart && (!S.selEntry || S.selEntry === n);
    const content = [];
    if (!e.on) {
      content.push(h("div", { class: "muted" }, "корневая деталь (ось " + JSON.stringify(e.axis || [0, 1, 0]) + ")"));
    } else {
      const at = e.at || {};
      const nr = partRounds(e.on);
      const isNum = typeof at.round === "number";
      content.push(h("div", { class: "muted" }, `пришита к: ${e.on}` + (e.applique ? " · аппликация" : "")));
      content.push(selectFld("Ряд", isNum ? "num" : (at.round ?? "num"), [["num", "число"], ["end", "конец (end)"], ["start", "начало (start)"]],
        v => { setOp(pre + "at.round", v === "num" ? Math.round(nr / 2) : v); queueOps([], true); }));
      if (isNum || at.round === undefined) content.push(slider("at.round", isNum ? at.round : nr / 2, 0, nr, 0.1, v => setOp(pre + "at.round", v)));
      content.push(slider("at.az °", at.az ?? 0, -180, 180, 1, v => setOp(pre + "at.az", v), { unset: at.az === undefined }));
      const d = e.dir || null, dcur = (d || [0, 0, 0]).slice();
      ["x", "y", "z"].forEach((ax, i) => content.push(slider(`dir ${ax}`, dcur[i], -1, 1, 0.05, v => {
        dcur[i] = v; setOp(pre + "dir", dcur.slice());
      }, { unset: !d })));
      content.push(slider("sink мм", e.sink ?? 0, 0, 20, 0.5, v => setOp(pre + "sink", v), { unset: e.sink === undefined }));
      if (e.roll !== undefined || e.applique) content.push(slider("roll °", e.roll ?? 0, -180, 180, 1, v => setOp(pre + "roll", v)));
      const cb = h("input", { type: "checkbox", checked: !!e.mirror });
      cb.addEventListener("change", () => setOp(pre + "mirror", cb.checked));
      content.push(h("div", { class: "fld wide" }, h("label", {}, "Зеркально"), h("label", {}, cb, " пара L/R")));
      if (e.avoid !== undefined || e.contact !== undefined) {
        const av = h("input", { type: "checkbox", checked: !!e.avoid });
        av.addEventListener("change", () => setOp(pre + "avoid", av.checked));
        content.push(h("div", { class: "fld wide" }, h("label", {}, "avoid"), h("label", {}, av, " отводить от других деталей")));
        content.push(slider("contact", e.contact ?? 0.3, 0, 1, 0.05, v => setOp(pre + "contact", v)));
      }
    }
    return card([h("b", {}, n), h("span", { class: "muted" }, e.part !== n ? `(${e.part})` : "")], sel,
      () => select(sel ? null : e.part, n), content, [h("span", { class: "info" }, e.mirror ? "L/R" : "")]);
  }));
}

// (c) face
function faceKey(f, i) { return f.id ?? String(i); }
function tabFace() {
  const items = S.spec.face || [];
  return h("div", {}, items.map((f, i) => {
    const key = faceKey(f, i), pre = `face.${key}.`;
    const sel = S.selFace === key;
    const nr = partRounds(f.on);
    const content = [h("div", { class: "muted" }, `${f.type} на ${f.on}` + (f.mirror ? " · зеркально" : ""))];
    if (f.at) {
      content.push(slider("Ряд", typeof f.at.round === "number" ? f.at.round : nr / 2, 0, nr, 0.1, v => setOp(pre + "at.round", v)));
      content.push(slider("az °", f.at.az ?? 0, -180, 180, 1, v => setOp(pre + "at.az", v)));
    }
    if (Array.isArray(f.size_mm)) {
      const scur = f.size_mm.slice();
      scur.forEach((s, j) => content.push(slider(j ? "высота мм" : "ширина мм", s, 1, 40, 0.5, v => {
        scur[j] = v; setOp(pre + "size_mm", scur.slice());
      })));
    } else if (f.size_mm !== undefined || /eye|blush/.test(f.type)) {
      content.push(slider("Размер мм", f.size_mm ?? 8, 1, 40, 0.5, v => setOp(pre + "size_mm", v)));
    }
    if (f.colour !== undefined) content.push(selectFld("Цвет", f.colour, colourOpts(false), v => setOp(pre + "colour", v)));
    if (f.points) {
      const t = h("input", { value: JSON.stringify(f.points), style: "width:100%" });
      t.addEventListener("change", () => { try { setOp(pre + "points", JSON.parse(t.value)); queueOps([], true); } catch { showErr("points: нужен JSON вида [[ряд, az], …]"); } });
      content.push(h("div", { class: "fld wide" }, h("label", {}, "Точки [ряд, az]"), t));
    }
    content.push(h("div", { class: "row" }, h("span", { style: "flex:1" }),
      h("button", { class: "danger", onclick: () => queueOps([{ op: "remove", path: "face." + key }], true) }, "Удалить")));
    return card([h("b", {}, f.id || f.type)], sel, () => { S.selFace = sel ? null : key; renderPanel(); }, content);
  }));
}

// (d) colours
function tabColours() {
  const cols = S.spec.colours || {};
  const rows = Object.entries(cols).map(([k, v]) => {
    const inp = h("input", { type: "color", value: /^#[0-9a-f]{6}$/i.test(v) ? v : "#888888" });
    const hex = h("span", { class: "muted" }, v);
    inp.addEventListener("input", () => { hex.textContent = inp.value; setOp("colours." + k, inp.value); });
    const used = (S.spec.parts || []).filter(p => p.colour === k).map(p => p.id).join(", ");
    return h("div", { class: "colrow" }, inp, h("span", { class: "nm" }, k), hex, h("span", { class: "muted" }, used ? "· " + used : ""));
  });
  const nm = h("input", { placeholder: "имя (pink)", size: 10 });
  const cv = h("input", { type: "color", value: "#e8a0a8" });
  return h("div", {}, rows,
    h("div", { class: "colrow", style: "margin-top:14px" }, cv, nm,
      h("button", { onclick: () => {
        const k = nm.value.trim();
        if (!/^[A-Za-z_][\w-]*$/.test(k)) { showErr("имя цвета: латиница"); return; }
        queueOps([{ op: "set", path: "colours." + k, value: cv.value }], true);
      } }, "Добавить цвет")));
}

// (e) JSON
function tabJson() {
  const ta = h("textarea", { class: "json", spellcheck: "false" }, JSON.stringify(S.spec, null, 2));
  return h("div", {}, ta, h("div", { class: "row" },
    h("button", { class: "primary", onclick: async () => {
      let spec;
      try { spec = JSON.parse(ta.value); } catch (e) { showErr("JSON: " + e.message); return; }
      await commitSpec(spec);
    } }, "Применить JSON"),
    h("button", { onclick: () => renderPanel() }, "Сбросить"),
    h("span", { class: "muted" }, "сводка: "), h("button", { onclick: () => openModal(h("pre", {}, S.summary)) }, "summary")));
}

// ------------------------------------------------------------------------------------- LLM command box
$("#cmdBtn").addEventListener("click", runCommand);
$("#cmdText").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runCommand(); });
async function runCommand() {
  const text = $("#cmdText").value.trim();
  if (!text || !S.spec) return;
  $("#cmdBtn").disabled = true; $("#cmdStatus").textContent = "думаю…"; $("#cmdResult").classList.add("hidden");
  try {
    const res = await api("/api/command", { spec: S.spec, text });
    S.cmd = res;
    $("#cmdExplain").textContent = res.explanation || "(без пояснения)";
    $("#cmdOps").textContent = res.ops.map(o => JSON.stringify(o)).join("\n") || "(нет операций)";
    $("#cmdResult").classList.remove("hidden");
    $("#cmdStatus").textContent = res.report?.flags?.length ? `флагов: ${res.report.flags.length}` : "";
  } catch (e) {
    $("#cmdStatus").textContent = e.status === 503 ? "LLM недоступна: " + e.message : "Ошибка: " + e.message;
  } finally { $("#cmdBtn").disabled = false; }
}
$("#cmdApply").addEventListener("click", async () => {
  if (!S.cmd) return;
  $("#cmdResult").classList.add("hidden");
  if (await commitSpec(S.cmd.spec)) { $("#cmdStatus").textContent = "применено"; $("#cmdText").value = ""; }
  S.cmd = null;
});
$("#cmdCancel").addEventListener("click", () => { S.cmd = null; $("#cmdResult").classList.add("hidden"); $("#cmdStatus").textContent = ""; });

// ------------------------------------------------------------------------------------- header actions
function openModal(content) { $("#modalBody").replaceChildren(content); $("#modal").classList.remove("hidden"); }
$("#modalClose").addEventListener("click", () => $("#modal").classList.add("hidden"));
$("#modal").addEventListener("click", e => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });

$("#undoBtn").addEventListener("click", () => stepHistory(-1));
$("#redoBtn").addEventListener("click", () => stepHistory(1));
document.addEventListener("keydown", e => {
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); stepHistory(e.shiftKey ? 1 : -1); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); stepHistory(1); }
  if (e.key === "Escape") $("#modal").classList.add("hidden");
});

$("#renderBtn").addEventListener("click", async () => {
  const b = $("#renderBtn"); b.disabled = true; setBusy(true, "рендер четырёх видов…");
  try {
    const r = await api("/api/render", { spec: S.spec });
    openModal(h("div", {}, h("div", { class: "muted" }, `рендер ${r.timings.total} с · флагов: ${r.flags.length}`), h("img", { src: r.png_url + "?t=" + Date.now() })));
  } catch (e) { showErr(e.message); }
  finally { b.disabled = false; setBusy(false); }
});
$("#patternBtn").addEventListener("click", async () => {
  const u = S.lastBuild?.pattern_url; if (!u) return;
  const t = await (await fetch(u)).text();
  openModal(h("pre", {}, t));
});
$("#saveBtn").addEventListener("click", async () => {
  const name = $("#saveName").value.trim() || (S.spec.name || "toy") + "_edit";
  try {
    const r = await api("/api/save", { name, spec: S.spec });
    $("#saveName").value = name;
    $("#status").textContent = "сохранено: out/editor/" + name + ".json"; $("#status").title = r.path;
    await loadToyList(r.id);
  } catch (e) { showErr("Сохранение: " + e.message); }
});

async function loadToyList(selectId) {
  const toys = await api("/api/toys");
  const sel = $("#toySelect");
  sel.replaceChildren(...toys.map(t => h("option", { value: t.id, selected: t.id === (selectId || S.toyId) },
    (t.source === "saved" ? "💾 " : "") + t.id + (t.title ? " — " + t.title : ""))));
  return toys;
}
$("#toySelect").addEventListener("change", e => openToy(e.target.value));

async function openToy(id) {
  const t = await api("/api/toy/" + id);
  S.toyId = id; S.history = []; S.hi = -1; S.selPart = null; S.selFace = null; firstFit = true;
  S.summary = t.summary; S.info = parseSummary(t.summary);
  $("#saveName").value = id.startsWith("saved/") ? id.slice(6) : id + "_edit";
  S.spec = t.spec;
  pushHistory(S.spec);
  renderPanel();
  await rebuild();
}

(async () => {
  const toys = await loadToyList();
  const first = new URLSearchParams(location.search).get("toy") || (toys.find(t => t.id === "teddy") || toys[0])?.id;
  if (first) { await loadToyList(first); await openToy(first); }
})();
window.__amikit = S; // for debugging in the console
