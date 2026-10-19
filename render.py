"""Studio renderer (moderngl, OpenGL 3.3): soft key light with shadow, ambient occlusion baked per vertex from many
directions, yarn sheen and groove darkening, glossy safety eyes, a soft floor. Renders the four views used by
the concept turntables (front, 3/4, side, back).
"""
from __future__ import annotations
import numpy as np
import moderngl
from PIL import Image

VIEWS = {"front": 0.0, "34": 35.0, "side": 90.0, "back": 180.0}
# measured by calibrate.py (input grey -> rendered lit-front grey, sRGB); its inverse is applied to colours
CAL_IN = np.array([0.0, 0.05, 0.169, 0.287, 0.406, 0.525, 0.644, 0.762, 0.881, 1.0])
# (re-measured 2026-10-09 after the per-pixel stitch relief; the 2026-10-08 softer light had left the old curve
# 5-12 % too dark in the mid-tones, so colours rendered lighter / paler than entered)
CAL_OUT = np.array([0.089, 0.103, 0.201, 0.324, 0.455, 0.582, 0.710, 0.838, 0.953, 0.990])

_VS = """
#version 330
uniform mat4 mvp;
uniform mat4 light_mvp;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in float in_ao; in float in_h; in float in_mat;
in vec3 in_nb; in vec2 in_uv; in float in_amp;
out vec3 v_pos; out vec3 v_nrm; out vec3 v_col; out float v_ao; out float v_h; out float v_mat; out vec4 v_ls;
out vec3 v_nb; out vec2 v_uv; out float v_amp;
void main(){
  v_pos=in_pos; v_nrm=in_nrm; v_col=in_col; v_ao=in_ao; v_h=in_h; v_mat=in_mat;
  v_nb=in_nb; v_uv=in_uv; v_amp=in_amp;
  v_ls = light_mvp*vec4(in_pos,1.0);
  gl_Position = mvp*vec4(in_pos,1.0);
}
"""

_FS = """
#version 330
uniform vec3 cam; uniform vec3 key_dir; uniform vec3 fill_dir; uniform vec3 rim_dir;
uniform sampler2D shadow_map; uniform float shadow_texel; uniform float exposure; uniform vec3 bg_col; uniform float floor_fade;
uniform float transparent; uniform vec2 floor_c; uniform float shadow_r; uniform float frame_px;
in vec3 v_pos; in vec3 v_nrm; in vec3 v_col; in float v_ao; in float v_h; in float v_mat; in vec4 v_ls;
in vec3 v_nb; in vec2 v_uv; in float v_amp;
out vec4 frag;
STITCH_GLSL
vec3 to_lin(vec3 c){ return pow(c, vec3(2.2)); }
float shadow(){
  vec3 p = v_ls.xyz / v_ls.w * 0.5 + 0.5;
  if(p.x<0.0||p.x>1.0||p.y<0.0||p.y>1.0||p.z>1.0) return 1.0;
  // fade to lit towards the map's border: past it nothing is known, and a hard cut there draws a straight
  // edge across the floor (the toy itself never reaches the border)
  float edge = 1.0 - smoothstep(0.8, 0.98, max(abs(p.x*2.0-1.0), abs(p.y*2.0-1.0)));
  float s=0.0; float n=0.0;
  for(int i=-3;i<=3;i++) for(int j=-3;j<=3;j++){
    float d = texture(shadow_map, p.xy + vec2(i,j)*shadow_texel*1.6).r;
    s += (p.z - 0.0015 <= d) ? 1.0 : 0.0; n += 1.0;
  }
  return mix(1.0, s/n, edge);
}
void main(){
  vec3 N = normalize(v_nrm);
  vec3 V = normalize(cam - v_pos);
  float hpx = v_h;
  if(v_amp > 0.0 && v_mat < 0.5){
    // the stitch relief drawn per pixel on the surface under it (the same V legs the mesh carries, at full
    // resolution): bump mapping from the height's screen-space gradient (Mikkelsen 2010)
    vec3 Nb = normalize(v_nb);
    hpx = stitch_h(fract(v_uv.x), clamp(fract(v_uv.y + 1e-4), 0.0, 1.0));
    float hw = v_amp * (hpx - 0.45);
    vec3 dpdx = dFdx(v_pos), dpdy = dFdy(v_pos);
    float dhdx = dFdx(hw), dhdy = dFdy(hw);
    vec3 r1 = cross(dpdy, Nb), r2 = cross(Nb, dpdx);
    float det = dot(dpdx, r1);
    vec3 grad = sign(det) * (dhdx * r1 + dhdy * r2);
    vec3 Nn = normalize(abs(det) * Nb - grad);
    if(abs(det) > 1e-12) N = Nn;
  }
  if(dot(N,V) < 0.0 && v_mat < 3.5) N = -N;
  vec3 alb = to_lin(v_col);
  float sh = mix(0.56, 1.0, shadow());   // big soft boxes: shadows are soft and never black (photo-studio look)
  float ao = v_ao;
  vec3 col;
  vec3 keyc = vec3(1.0,0.97,0.92)*2.3; vec3 fillc = vec3(0.92,0.92,1.0)*0.70; vec3 rimc = vec3(1.0)*0.7;
  vec3 sky = vec3(0.92,0.94,1.0)*0.55; vec3 gnd = vec3(0.84,0.80,0.76)*1.02;   // floor bounce (studio fill from below: white under a chin stays white)
  vec3 amb = mix(gnd, sky, N.y*0.5+0.5);
  if(v_mat < 0.5 || v_mat > 2.5){
    // yarn / felt: wrapped diffuse, groove darkening, fuzzy sheen
    float cav = (v_mat < 0.5) ? mix(0.70, 1.0, smoothstep(0.0, 0.8, hpx)) : 1.0;
    float wrap = 0.5;
    float dk = max((dot(N,key_dir)+wrap)/(1.0+wrap), 0.0);
    float df = max((dot(N,fill_dir)+wrap)/(1.0+wrap), 0.0);
    float fres = pow(1.0 - max(dot(N,V),0.0), 3.0);
    float rim = pow(max(dot(-V, rim_dir)*0.5+0.5,0.0),2.0) * fres;
    // studio bounce cards light the undersides too: ambient only partly occluded (soft creases, no dark ring under a head)
    col = alb * (keyc*dk*sh + fillc*df + amb*mix(1.0, ao, 0.6)*1.2) * cav;
    col += (alb*0.85+0.15*vec3(1.0)) * (fres*0.06*ao + rim*rimc*0.30*ao);
    vec3 H = normalize(key_dir+V);
    col += keyc * 0.025 * pow(max(dot(N,H),0.0), 12.0) * sh * cav;
    col *= mix(1.0, ao, 0.15);
  } else if(v_mat < 1.5){
    // glossy plastic (safety eyes, noses)
    float dk = max(dot(N,key_dir),0.0);
    col = alb * (keyc*dk*sh*0.6 + amb*ao);
    vec3 R = reflect(-V, N);
    // one main catchlight (the key; printed glints sit there too) and a faint fill glint: two equal sharp
    // reflections on opposite sides read as noise, not as a glossy eye
    float spec = pow(max(dot(R,key_dir),0.0), 180.0)*9.0*sh + pow(max(dot(R,fill_dir),0.0), 90.0)*0.35;
    float soft = smoothstep(0.80, 0.97, dot(R, normalize(key_dir+vec3(0.0,0.3,0.0))))*0.6*sh;
    float env = 0.015 + 0.35*pow(1.0-max(dot(N,V),0.0),5.0);
    col += vec3(1.0)*(spec + soft*0.5) + mix(gnd,sky,R.y*0.5+0.5)*env*ao*1.5;
  } else {
    // embroidery thread: smoother than yarn, a little sheen
    float dk = max((dot(N,key_dir)+0.2)/1.2,0.0);
    col = alb*(keyc*dk*sh + amb*ao*1.2);
    vec3 H = normalize(key_dir+V);
    col += keyc*0.08*pow(max(dot(N,H),0.0),30.0)*sh;
  }
  if(v_mat > 3.5){
    // floor: receives shadow and AO only, fades into the backdrop
    float d = length(v_pos.xz);
    // a soft studio floor: the key's cast shadow is faint (big soft boxes), the contact shadow (AO) carries it
    float lit = (0.80 + 0.20*sh) * (0.42 + 0.58*ao);
    float fade = smoothstep(floor_fade*0.5, floor_fade, d);
    if(transparent > 0.5){
      // transparent preview: the floor is only its contact shadow, a warm dark veil over whatever is behind
      // the faint AO far out on the floor would read as a grey veil: keep only the shadow proper
      // and it falls off radially to nothing well inside the floor and the frame, so a tile never shows its edge
      float fall = 1.0 - smoothstep(shadow_r*0.45, shadow_r, length(v_pos.xz - floor_c));
      // and to nothing at the frame's border too (a tall toy's frame cuts the floor in front of it)
      vec2 fc = min(gl_FragCoord.xy, vec2(frame_px) - gl_FragCoord.xy);
      fall *= smoothstep(0.0, frame_px*0.08, min(fc.x, fc.y));
      float a = clamp(((1.0 - lit) - 0.05) * fall * 1.25, 0.0, 0.85);
      frag = vec4(0.17, 0.11, 0.09, a);
      return;
    }
    vec3 fl = to_lin(bg_col) * lit * 1.0;
    col = mix(fl, to_lin(bg_col), fade);
  }
  // filmic-ish tone map; exposure calibrated so a lit front surface shows the yarn's photo colour
  if(v_mat < 3.5) col *= exposure;
  // linear up to 0.8, then a soft shoulder: yarn colours come out as entered (see calibrate.py)
  vec3 hi = 0.8 + 0.2*(1.0 - exp(-(col-0.8)/0.2));
  col = mix(col, hi, step(0.8, col));
  frag = vec4(pow(clamp(col,0.0,1.0), vec3(1.0/2.2)), 1.0);
}
"""

STITCH_GLSL = """
// piece.stitch_height in GLSL (keep the two in step): relief of one sc stitch, 0 groove .. 1 top of a yarn leg
float stitch_h(float fu, float fv){
  float h = 0.0;
  for(int k = 0; k < 2; k++){
    float sg = (k == 0) ? -1.0 : 1.0;
    vec2 p0 = vec2(0.5 + sg * 0.08, 0.14);
    vec2 p1 = vec2(0.5 + sg * 0.34, 0.86);
    vec2 d = p1 - p0;
    float L2 = dot(d, d);
    float px = fu - p0.x, py = (fv - p0.y) * 0.95;
    float s = clamp((px * d.x + py * d.y * 0.95) / L2, 0.0, 1.0);
    float qx = px - s * d.x, qy = py - s * d.y * 0.95;
    float dist = sqrt(qx * qx + qy * qy);
    float leg = clamp(1.0 - (dist / 0.285) * (dist / 0.285), 0.0, 1.0);
    h = max(h, pow(leg, 0.42) * (0.80 + 0.20 * sin(3.14159265 * s)));
  }
  float env = pow(sin(3.14159265 * clamp(fv, 0.0, 1.0)), 0.5);
  return 0.36 * (1.0 - env) + h * env + 0.36 * env * (1.0 - h) * 0.6;
}
"""
_FS = _FS.replace("STITCH_GLSL", STITCH_GLSL)

_DEPTH_VS = """
#version 330
uniform mat4 mvp; in vec3 in_pos;
void main(){ gl_Position = mvp*vec4(in_pos,1.0); }
"""
_DEPTH_FS = """
#version 330
void main(){}
"""


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fov_deg, aspect, near, far):
    f = 1 / np.tan(np.radians(fov_deg) / 2)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect; M[1, 1] = f
    M[2, 2] = (far + near) / (near - far); M[2, 3] = 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


def _box_blur(a, w, passes=3):
    """Near-gaussian blur of a 2-D array (`passes` box filters of odd width `w`, edges clamped). Smooths the
    floor's per-vertex AO: a few dozen hard per-direction occlusions otherwise draw radial streaks into the contact
    shadow. On the floor's grid (dense under the toy) it softens more the further out, like a real penumbra."""
    h = w // 2
    for _ in range(passes):
        for ax in (0, 1):
            pad = [(0, 0), (0, 0)]; pad[ax] = (h + 1, h)
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
            a = (np.take(c, range(w, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - w), axis=ax)) / w
    return a


def ortho(half, near, far):
    M = np.eye(4)
    M[0, 0] = M[1, 1] = 1 / half
    M[2, 2] = -2 / (far - near); M[2, 3] = -(far + near) / (far - near)
    return M


def _gl(m):
    return m.T.astype("f4").tobytes()


class Renderer:
    def __init__(self, size=900, ssaa=2):
        self.ctx = moderngl.create_standalone_context()
        self.size = size
        self.ss = ssaa
        self.use_calibration = True      # calibrate.py measures the raw transfer curve with this off
        self.bump = 1.0                  # per-pixel stitch relief strength (0 = the mesh's own relief only)
        self.exposure = 0.52
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self.dprog = self.ctx.program(vertex_shader=_DEPTH_VS, fragment_shader=_DEPTH_FS)
        self.ctx.enable(moderngl.DEPTH_TEST)

    # scene ------------------------------------------------------------------------------------------------
    def set_scene(self, parts, floor=True, ao_dirs=96):
        V = []; N = []; C = []; H = []; M = []; F = []; off = 0
        NB = []; UV = []; AMP = []
        for p in parts:
            V.append(p["V"]); N.append(p["N"]); C.append(p["C"]); H.append(p["H"])
            M.append(np.full(len(p["V"]), float(p["mat"]))); F.append(p["F"] + off); off += len(p["V"])
            # per-pixel stitch relief (yarn pieces made by toy._mesh_piece): base normal, stitch coords, amplitude
            nb = p.get("NB")
            NB.append(nb if nb is not None else p["N"])
            UV.append(p["UV"] if (nb is not None and "UV" in p) else np.zeros((len(p["V"]), 2)))
            AMP.append(p["AMP"] * self.bump if (nb is not None and "AMP" in p) else np.zeros(len(p["V"])))
        V = np.concatenate(V); N = np.concatenate(N)
        lo, hi = V.min(0), V.max(0)
        self.center = (lo + hi) / 2
        self.radius = float(np.linalg.norm(hi - lo) / 2)
        n_obj = len(V)
        self._floor_ng = 0
        if floor:
            ext = self.radius * 6
            # dense under the toy, where the contact shadow (per-vertex AO) lives, sparse out to the horizon
            ng = 241
            t = np.linspace(-1, 1, ng)
            g = ext * np.sign(t) * t ** 2
            GX, GZ = np.meshgrid(g + self.center[0], g + self.center[2], indexing="ij")
            gv = np.stack([GX, np.zeros_like(GX) - 0.05, GZ], -1).reshape(-1, 3)
            ii = np.arange(ng * ng).reshape(ng, ng)
            self._floor_ng = ng
            gf = np.concatenate([np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:]], -1).reshape(-1, 3),
                                 np.stack([ii[:-1, :-1], ii[1:, 1:], ii[1:, :-1]], -1).reshape(-1, 3)])
            NB.append(np.tile([0, 1.0, 0], (len(gv), 1))); UV.append(np.zeros((len(gv), 2))); AMP.append(np.zeros(len(gv)))
            V = np.vstack([V, gv]); N = np.vstack([N, np.tile([0, 1.0, 0], (len(gv), 1))])
            C.append(np.tile([0.9, 0.9, 0.9], (len(gv), 1))); H.append(np.ones(len(gv))); M.append(np.full(len(gv), 4.0))
            F.append(gf + off)
        self.V = V.astype("f4"); self.N = N.astype("f4")
        self.NB = np.concatenate(NB).astype("f4"); self.UV = np.concatenate(UV).astype("f4")
        self.AMP = np.concatenate(AMP).astype("f4")
        C = np.concatenate(C)
        # inverse of the measured transfer curve (calibrate.py): dark yarns otherwise come out lifted by the
        # studio's ambient and sheen; applied to everything but the floor
        Mcat = np.concatenate(M)
        obj = Mcat < 3.5
        if self.use_calibration:
            C[obj] = np.interp(C[obj], CAL_OUT, CAL_IN)
        self.C = C.astype("f4"); self.H = np.concatenate(H).astype("f4")
        self.M = np.concatenate(M).astype("f4"); self.F = np.concatenate(F).astype("i4")
        self.n_obj = n_obj
        self.F_obj = np.concatenate(F[:len(parts)]).astype("i4")
        self.AO = self._bake_ao(ao_dirs).astype("f4")
        if self._floor_ng:
            ng = self._floor_ng
            self.AO[n_obj:] = _box_blur(self.AO[n_obj:].reshape(ng, ng), 9).ravel()
        self._upload()

    def release(self):
        for k in ("vao", "vbo", "ibo", "vao_d", "vbo_d", "ibo_d"):
            o = getattr(self, k, None)
            if o is not None:
                o.release()
                setattr(self, k, None)

    def _upload(self):
        ctx = self.ctx
        self.release()
        data = np.hstack([self.V, self.N, self.C, self.AO[:, None], self.H[:, None], self.M[:, None],
                          self.NB, self.UV, self.AMP[:, None]]).astype("f4")
        self.vbo = ctx.buffer(data.tobytes())
        self.ibo = ctx.buffer(self.F.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 3f 1f 1f 1f 3f 2f 1f",
                                                 "in_pos", "in_nrm", "in_col", "in_ao", "in_h", "in_mat",
                                                 "in_nb", "in_uv", "in_amp")], self.ibo)
        self.vbo_d = ctx.buffer(self.V.tobytes())
        self.ibo_d = ctx.buffer(self.F_obj.tobytes())
        self.vao_d = ctx.vertex_array(self.dprog, [(self.vbo_d, "3f", "in_pos")], self.ibo_d)

    def _depth_view(self, d, res):
        """Orthographic depth map of the toy seen from direction d (looking along -d)."""
        c = self.center; R = self.radius * 1.15
        eye = c + d * R * 3
        up = (0, 1, 0) if abs(d[1]) < 0.95 else (0, 0, 1)
        view = look_at(eye, c, up)
        proj = ortho(R, R * 1.0, R * 5.0)
        mvp = proj @ view
        if not hasattr(self, "_dfbo") or self._dfbo_res != res:
            self._dtex = self.ctx.depth_texture((res, res))
            self._dfbo = self.ctx.framebuffer(depth_attachment=self._dtex)
            self._dfbo_res = res
        self._dfbo.use(); self._dfbo.clear(depth=1.0)
        self.dprog["mvp"].write(_gl(mvp))
        self.vao_d.render(moderngl.TRIANGLES)
        depth = np.frombuffer(self._dtex.read(alignment=1), dtype="f4").reshape(res, res)
        return mvp, depth

    def _bake_ao(self, n_dirs, res=1024):
        # temporary upload for depth renders
        self.release()
        self.vbo_d = self.ctx.buffer(self.V.tobytes())
        self.ibo_d = self.ctx.buffer(self.F_obj.tobytes())
        self.vao_d = self.ctx.vertex_array(self.dprog, [(self.vbo_d, "3f", "in_pos")], self.ibo_d)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Fibonacci sphere directions
        k = np.arange(n_dirs) + 0.5
        phi = np.arccos(1 - 2 * k / n_dirs); th = np.pi * (1 + 5 ** 0.5) * k
        dirs = np.stack([np.sin(phi) * np.cos(th), np.cos(phi), np.sin(phi) * np.sin(th)], -1)
        P = self.V.astype(float) + self.N.astype(float) * 0.35
        Nn = self.N.astype(float)
        is_floor = self.M > 3.5
        num = np.zeros(len(P)); den = np.zeros(len(P))
        for d in dirs:
            w = np.clip(Nn @ d, 0, None)
            if d[1] < -0.02:
                den += w           # the floor blocks everything below the horizon
                continue
            mvp, depth = self._depth_view(d, res)
            ph = np.hstack([P, np.ones((len(P), 1))]) @ mvp.T
            ndc = ph[:, :3] / ph[:, 3:4]
            x = np.clip(((ndc[:, 0] * 0.5 + 0.5) * res).astype(int), 0, res - 1)
            y = np.clip(((ndc[:, 1] * 0.5 + 0.5) * res).astype(int), 0, res - 1)
            z = ndc[:, 2] * 0.5 + 0.5
            # outside this depth view nothing of the toy is in the way (clamping to the edge pixel would read
            # the toy's own border depth there: blocky false occlusion over the far floor, cut by straight edges)
            out = (np.abs(ndc[:, 0]) > 1) | (np.abs(ndc[:, 1]) > 1)
            vis = ((z - 0.0008 <= depth[y, x]) | out).astype(float)
            num += w * vis; den += w
        ao = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0)
        return np.clip(ao, 0, 1) ** 0.8

    # render -----------------------------------------------------------------------------------------------
    def render(self, az_deg=0.0, elev_deg=8.0, fov=26.0, bg="#d4ccc9", fit=1.0, target=None, radius=None,
               transparent=False):
        """target / radius (optional): frame a part of the scene (a head close-up) instead of the whole toy.
        transparent: RGBA image - the toy opaque, the backdrop clear, the floor reduced to its soft contact
        shadow (semi-transparent), for site previews over any colour. Same lighting and colours as the studio
        render; the backdrop colour still tints the ambient bounce via `bg`."""
        S = self.size * self.ss
        ctx = self.ctx
        tc = self.center.copy() if target is None else np.asarray(target, float)
        rad = self.radius if radius is None else float(radius)
        a = np.radians(az_deg); e = np.radians(elev_deg)
        fwd = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        dist = rad * fit / np.tan(np.radians(fov) / 2) * 1.05
        eye = tc + fwd * dist
        far = dist * 4 if radius is None else dist * 4 + self.radius * 2
        view = look_at(eye, tc); proj = perspective(fov, 1.0, dist * 0.2, far)
        c = self.center.copy()
        mvp = proj @ view
        # lights in camera space: key upper-left-front, fill right, rim behind
        right = np.cross(-fwd, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.array([0, 1.0, 0])

        def cam_dir(x, y, z):
            v = x * right + y * up + z * fwd
            return v / np.linalg.norm(v)
        key = cam_dir(-0.55, 0.75, 0.6); fill = cam_dir(0.8, 0.15, 0.5); rim = cam_dir(0.3, 0.5, -0.9)
        # shadow map from the key light
        res = 2048
        R = self.radius * 1.4
        lview = look_at(c + key * R * 3, c, (0, 1, 0) if abs(key[1]) < 0.95 else (0, 0, 1))
        lmvp = ortho(R * 1.2, R * 0.5, R * 6) @ lview
        stex = ctx.depth_texture((res, res)); sfbo = ctx.framebuffer(depth_attachment=stex)
        sfbo.use(); sfbo.clear(depth=1.0)
        self.dprog["mvp"].write(_gl(lmvp)); self.vao_d.render(moderngl.TRIANGLES)
        # main pass
        col_rb = ctx.renderbuffer((S, S)); dep_rb = ctx.depth_renderbuffer((S, S))
        fbo = ctx.framebuffer(color_attachments=[col_rb], depth_attachment=dep_rb)
        fbo.use()
        bgc = np.array([int(bg[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
        fbo.clear(*bgc, 0.0 if transparent else 1.0)
        stex.use(0)
        p = self.prog
        p["mvp"].write(_gl(mvp)); p["light_mvp"].write(_gl(lmvp)); p["cam"].value = tuple(eye)
        p["key_dir"].value = tuple(key); p["fill_dir"].value = tuple(fill); p["rim_dir"].value = tuple(rim)
        p["shadow_map"].value = 0; p["shadow_texel"].value = 1.0 / res
        p["bg_col"].value = tuple(bgc); p["exposure"].value = float(self.exposure); p["floor_fade"].value = float(self.radius * 3.2)
        p["transparent"].value = 1.0 if transparent else 0.0
        p["floor_c"].value = (float(self.center[0]), float(self.center[2])); p["shadow_r"].value = float(self.radius * 1.5)
        p["frame_px"].value = float(S)
        self.vao.render(moderngl.TRIANGLES)
        if transparent:
            img = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape(S, S, 4)[::-1]
            # downsample premultiplied, or the clear backdrop bleeds dark fringes into the edges
            im = Image.fromarray(img, "RGBA").convert("RGBa")
            if self.ss > 1:
                im = im.resize((self.size, self.size), Image.LANCZOS)
            im = im.convert("RGBA")
        else:
            img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(S, S, 3)[::-1]
            im = Image.fromarray(img)
            if self.ss > 1:
                im = im.resize((self.size, self.size), Image.LANCZOS)
        for o in (fbo, col_rb, dep_rb, sfbo, stex):
            o.release()
        return im

    def views(self, out_prefix=None, which=("front", "34", "side", "back"), **kw):
        ims = {}
        for v in which:
            ims[v] = self.render(VIEWS[v], **kw)
            if out_prefix:
                ims[v].save(f"{out_prefix}{v}.png")
        return ims


def sheet(images, labels=None, ref=None, size=420):
    """Horizontal sheet; optional reference image first."""
    tiles = []
    if ref is not None:
        r = ref.copy(); r.thumbnail((size * 2, size))
        tiles.append(r)
    for im in images:
        t = im.copy(); t.thumbnail((size, size)); tiles.append(t)
    W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
    out = Image.new("RGB", (W, size), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, (size - t.height) // 2)); x += t.width + 8
    return out
