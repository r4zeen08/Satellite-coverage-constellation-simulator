"""
Satellite Constellation Coverage Simulator (Python / VPython)
================================================================
pip install vpython pillow numpy

Run with:  python satellite_coverage_simulation.py
This opens a browser tab (VPython renders via a local GlowScript/WebGL view).

Controls
  - Drag empty space to orbit the camera, scroll to zoom (native VPython nav).
  - Click a satellite, then drag: it moves along its own orbital plane
    (phase angle changes; the plane, altitude and inclination stay fixed).
  - Sliders: satellite counts / altitudes / inclinations per shell,
    minimum elevation angle (controls coverage cone size), time scale.

This is a direct architectural port of the three.js/WebGL version, using the
same physics and the same population approximation. See inline notes for
where to plug in a real gridded population dataset (e.g. NASA SEDAC GPWv4).
"""

import math
import os
import numpy as np
from PIL import Image, ImageDraw
import vpython as vp

# ============================== CONSTANTS ==================================
MU_EARTH = 398600.4418        # km^3/s^2 (GM of Earth)
R_EARTH_KM = 6371.0
SCENE_R = 1.0                 # Earth radius in scene units
KM_TO_SCENE = SCENE_R / R_EARTH_KM
SIDEREAL_DAY_S = 86164.1

# ============================ POPULATION MODEL ==============================
# Same 55-region gaussian approximation as the JS version: [lat, lon, weight_millions, spread_deg]
POP_REGIONS = [
    [31,121,121,8],[39,116,116,7],[23,113,110,6],[30,104,90,7],
    [28,77,300,9],[19,73,120,6],[13,78,150,7],[23,89,250,7],
    [28,69,200,7],[-6,110,140,6],[0,120,80,10],[13,122,110,6],
    [16,106,95,5],[15,101,65,5],[20,96,54,6],[36,138,125,6],
    [37,127,52,4],[9,8,210,7],[9,38,115,6],[27,31,105,4],
    [0,25,95,7],[-29,25,60,6],[-6,35,130,8],[8,-2,180,8],
    [32,3,100,7],[5,20,90,8],[49,7,180,6],[54,-3,70,4],
    [41,5,120,6],[50,25,140,7],[56,38,90,6],[60,90,35,15],
    [39,35,85,5],[32,53,88,6],[40,-75,65,4],[33,-83,90,6],
    [41,-88,55,6],[30,-97,60,5],[36,-119,55,5],[19,-99,130,6],
    [14,-88,50,5],[-22,-46,110,6],[-8,-38,55,6],[-5,-60,45,9],
    [6,-70,95,7],[-10,-75,65,7],[-35,-65,60,7],[55,-97,38,12],
    [-25,134,26,12],[45,66,75,8],[24,45,60,6],[33,42,70,5],
    [27,85,55,4],[34,66,40,5],
]
BASELINE = 0.35
GRID_W, GRID_H = 180, 90
TOTAL_POP_MILLIONS = 7900.0

# Build lat/lon grid centers once (vectorized)
_lat_centers = 90 - (np.arange(GRID_H) + 0.5) * (180.0 / GRID_H)          # (H,)
_lon_centers = -180 + (np.arange(GRID_W) + 0.5) * (360.0 / GRID_W)        # (W,)
LON_GRID, LAT_GRID = np.meshgrid(_lon_centers, _lat_centers)               # (H,W)

def build_density_grid():
    """Vectorized gaussian-bump population model, normalized to TOTAL_POP_MILLIONS."""
    lat_r = np.radians(LAT_GRID)
    grid = np.full((GRID_H, GRID_W), BASELINE)
    for (rlat, rlon, pop, sigma) in POP_REGIONS:
        rlat_r, rlon_r = math.radians(rlat), math.radians(rlon)
        dlon = np.radians(LON_GRID - rlon)
        dlat = lat_r - rlat_r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * math.cos(rlat_r) * np.sin(dlon / 2) ** 2
        d_deg = np.degrees(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        grid += pop * np.exp(-(d_deg ** 2) / (2 * sigma ** 2))
    grid *= TOTAL_POP_MILLIONS / grid.sum()
    return grid

DENSITY = build_density_grid()   # (GRID_H, GRID_W), units = millions of people per cell

# Precompute unit vectors for every grid cell (for fast angular-distance checks)
def lat_lon_to_unit(lat_deg, lon_deg):
    phi = np.radians(90 - lat_deg)
    theta = np.radians(lon_deg + 180)
    x = -np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)
    return np.stack([x, y, z], axis=-1)

GRID_UNIT_VECS = lat_lon_to_unit(LAT_GRID, LON_GRID).reshape(-1, 3)   # (H*W, 3)
GRID_WEIGHTS = DENSITY.reshape(-1)                                     # (H*W,)

def ang_dist_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl, dp = math.radians(lon2 - lon1), p2 - p1
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return math.degrees(2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# =========================== EARTH TEXTURE (PIL) ============================
# Rough continent silhouettes (same simplified polygons as the JS version).
CONTINENTS = [
    [(71,-10),(70,60),(45,60),(35,45),(12,43),(8,45),(-2,40),(-35,18),(-34,20),
     (10,-17),(36,-6),(43,10),(55,10),(60,30),(71,-10)],
    [(55,60),(60,90),(70,140),(55,165),(42,132),(30,120),(20,105),(10,105),
     (8,98),(20,72),(35,60),(45,50),(55,60)],
    [(70,-165),(70,-60),(45,-52),(25,-80),(20,-105),(32,-117),(48,-124),(60,-140),(70,-165)],
    [(12,-70),(-5,-35),(-34,-58),(-55,-68),(-20,-70),(-2,-80),(12,-70)],
    [(-11,113),(-12,141),(-25,153),(-38,145),(-35,117),(-20,115),(-11,113)],
]

def heat_color(t):
    t = max(0.0, min(1.0, t)) ** 0.42
    stops = [(10,20,40,0),(40,70,140,60),(80,180,210,140),(255,210,90,210),(255,90,60,255)]
    idx = min(3, int(t * 4)); f = t * 4 - idx
    a, b = stops[idx], stops[idx + 1]
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(4))

def build_earth_texture(path="earth_texture.png", w=1024, h=512, show_heat=True):
    img = Image.new("RGB", (w, h), (13, 51, 80))
    draw = ImageDraw.Draw(img)
    # simple vertical ocean gradient
    for y in range(h):
        t = abs(y - h / 2) / (h / 2)
        c = tuple(int(10 + t * 8 + v) for v in (0, 25, 45))
        draw.line([(0, y), (w, y)], fill=(c[0], 10 + int(t*8), 40 + int(t*15)))

    def to_xy(lat, lon):
        return ((lon + 180) / 360 * w, (90 - lat) / 180 * h)

    for poly in CONTINENTS:
        pts = [to_xy(lat, lon) for lat, lon in poly]
        draw.polygon(pts, fill=(31, 61, 44), outline=(44, 82, 64))

    if show_heat:
        max_v = DENSITY.max()
        heat = Image.new("RGBA", (GRID_W, GRID_H))
        for j in range(GRID_H):
            for i in range(GRID_W):
                r, g, b, a = heat_color(DENSITY[j, i] / max_v)
                heat.putpixel((i, j), (r, g, b, int(a * 0.85)))
        heat = heat.resize((w, h), Image.BILINEAR)
        img = Image.alpha_composite(img.convert("RGBA"), heat).convert("RGB")

    img.save(path)
    return path

TEXTURE_PATH = build_earth_texture(show_heat=True)

# ================================ SCENE ======================================
scene = vp.canvas(title="",
                   width=1100, height=700, background=vp.vector(0.02, 0.03, 0.05),
                   align='left')
scene.range = 3
scene.forward = vp.vector(-1, -0.4, -1)

# starfield
star_pts = []
for _ in range(1200):
    r = 12 + np.random.rand() * 6
    theta = np.random.rand() * 2 * math.pi
    phi = math.acos(2 * np.random.rand() - 1)
    star_pts.append(vp.vector(r*math.sin(phi)*math.cos(theta), r*math.cos(phi), r*math.sin(phi)*math.sin(theta)))
vp.points(pos=star_pts, radius=0.4, color=vp.color.white, opacity=0.6)

earth = vp.sphere(pos=vp.vector(0,0,0), radius=SCENE_R, texture=TEXTURE_PATH, shininess=0.3)
vp.sphere(pos=vp.vector(0,0,0), radius=SCENE_R*1.02, color=vp.color.cyan, opacity=0.05)

sun_light = vp.distant_light(direction=vp.vector(1, 0.5, 0.6), color=vp.color.white)

# ============================ ORBITAL MECHANICS ==============================
def period_for_alt(alt_km):
    a = R_EARTH_KM + alt_km
    return 2 * math.pi * math.sqrt(a**3 / MU_EARTH)   # seconds

COLORS = {"LEO": vp.color.cyan, "MEO": vp.vector(1, 0.7, 0.32), "GEO": vp.vector(1, 0.42, 0.42)}

class Satellite:
    def __init__(self, sat_type, alt_km, incl_deg, raan_deg, m0_deg):
        self.type = sat_type
        self.alt_km = alt_km
        self.incl = math.radians(incl_deg)
        self.raan = math.radians(raan_deg)
        self.m0 = math.radians(m0_deg)
        self.a = R_EARTH_KM + alt_km
        self.period = period_for_alt(alt_km)
        self.phase_override = None
        self.pos = vp.vector(0, 0, 0)
        self.sub_lat = 0.0
        self.sub_lon = 0.0
        # orbital-plane basis vectors (derived from inclination + RAAN)
        ci, si, cO, sO = math.cos(self.incl), math.sin(self.incl), math.cos(self.raan), math.sin(self.raan)
        self.bx = vp.vector(cO, 0, sO)
        self.by = vp.vector(-sO * si, ci, cO * si)

    def theta(self, sim_t):
        if self.phase_override is not None:
            return self.phase_override
        return self.m0 + (2 * math.pi / self.period) * sim_t

    def update(self, sim_t, earth_rot_deg):
        th = self.theta(sim_t)
        xp, yp = self.a * math.cos(th), self.a * math.sin(th)
        p3 = (self.bx * xp + self.by * yp) * KM_TO_SCENE
        self.pos = p3
        r = p3.mag
        lat = 90 - math.degrees(math.acos(max(-1, min(1, p3.y / r))))
        lon = math.degrees(math.atan2(p3.z, -p3.x)) - earth_rot_deg - 180
        lon = ((lon + 540) % 360) - 180
        self.sub_lat, self.sub_lon = lat, lon

    def coverage_half_angle_deg(self, min_el_deg):
        min_el = math.radians(min_el_deg)
        ratio = R_EARTH_KM / self.a
        lam = math.acos(ratio * math.cos(min_el)) - min_el
        return math.degrees(lam)

# ============================ STATE / UI VARIABLES ============================
state = {
    "leo_n": 18, "leo_alt": 550, "leo_incl": 53,
    "meo_n": 12, "meo_alt": 20200, "meo_incl": 55,
    "geo_n": 8,
    "min_el": 10,
    "time_scale_exp": 3,   # 10**exp multiplier
    "playing": True,
    "show_cones": True,
}

satellites = []
sat_objs = []     # (Satellite, vp.sphere, vp.cone or None)
ring_objs = []
sim_time_s = 0.0
earth_rotation_deg = 0.0
selected = {"sat": None, "dragging": False}


def clear_visuals():
    for _, mesh, cone in sat_objs:
        mesh.visible = False
        if cone: cone.visible = False
    for r in ring_objs:
        r.visible = False
    sat_objs.clear()
    ring_objs.clear()
    satellites.clear()


def make_ring(a_km, incl_deg, raan_deg, color):
    incl, raan = math.radians(incl_deg), math.radians(raan_deg)
    ci, si, cO, sO = math.cos(incl), math.sin(incl), math.cos(raan), math.sin(raan)
    normal = vp.vector(-sO*ci, -si, cO*ci)  # any vector normal to orbital plane
    return vp.ring(pos=vp.vector(0,0,0), axis=normal, radius=a_km*KM_TO_SCENE,
                   thickness=0.004, color=color, opacity=0.35)


def generate_constellation():
    clear_visuals()
    cfgs = [
        ("LEO", state["leo_n"], state["leo_alt"], state["leo_incl"]),
        ("MEO", state["meo_n"], state["meo_alt"], state["meo_incl"]),
        ("GEO", state["geo_n"], 35786, 0),
    ]
    for sat_type, n, alt, incl in cfgs:
        if n <= 0:
            continue
        planes = 1 if sat_type == "GEO" else max(1, min(6, math.ceil(math.sqrt(n))))
        per_plane = math.ceil(n / planes)
        created = 0
        for p in range(planes):
            if created >= n:
                break
            raan = (360 / planes) * p
            ring_objs.append(make_ring(R_EARTH_KM + alt, incl, raan, COLORS[sat_type]))
            for s in range(per_plane):
                if created >= n:
                    break
                m0 = (360 / per_plane) * s + (p * 17) % 360
                sat = Satellite(sat_type, alt, incl, raan, m0)
                satellites.append(sat)
                mesh = vp.sphere(radius=0.028, color=COLORS[sat_type], make_trail=False)
                mesh.sat_ref = sat
                cone = None
                if state["show_cones"]:
                    cone = vp.cone(radius=0.01, color=COLORS[sat_type], opacity=0.10)
                sat_objs.append((sat, mesh, cone))
                created += 1


# ================================ COVERAGE ===================================
coverage_stats = {"pct": 0.0, "covered_millions": 0.0, "total_millions": TOTAL_POP_MILLIONS}

def compute_coverage():
    if not satellites:
        coverage_stats["pct"] = 0.0
        coverage_stats["covered_millions"] = 0.0
        return
    covered_mask = np.zeros(GRID_UNIT_VECS.shape[0], dtype=bool)
    min_el = state["min_el"]
    for sat in satellites:
        sub_unit = lat_lon_to_unit(np.array([sat.sub_lat]), np.array([sat.sub_lon]))[0]
        cos_d = GRID_UNIT_VECS @ sub_unit
        ang = np.degrees(np.arccos(np.clip(cos_d, -1, 1)))
        half = sat.coverage_half_angle_deg(min_el)
        covered_mask |= (ang <= half)
    covered = GRID_WEIGHTS[covered_mask].sum()
    coverage_stats["covered_millions"] = covered
    coverage_stats["pct"] = 100 * covered / TOTAL_POP_MILLIONS


# ================================ UI WIDGETS =================================
# NOTE on VPython's caption HTML: every call to append_to_caption() is parsed
# and inserted as a *complete, self-closed* fragment — you can't open a <div>
# in one call and close it three calls later, the browser will auto-close it
# immediately. So every snippet below is fully self-contained; layout comes
# from a global <style> block plus consistent inline styling per call, not
# from nested containers.

CSS = """
<style>
  body { background:#0b0f16; color:#d7e2ec; font-family:'Segoe UI',Helvetica,Arial,sans-serif; }
  canvas { border-radius:10px; }
  .title    { font-size:18px; font-weight:700; color:#4fd6c0; letter-spacing:.04em; }
  .subtitle { font-size:12.5px; color:#8aa2b8; line-height:1.6; }
  .sec-h    { display:inline-block; margin:16px 0 8px; font-size:11px; font-weight:700;
              letter-spacing:.14em; text-transform:uppercase; color:#7c8ea3;
              border-top:1px solid #24344a; padding-top:12px; width:100%; }
  .lab      { font-size:13px; color:#d7e2ec; }
  .leo-c    { color:#4fd6c0; font-weight:600; }
  .meo-c    { color:#ffb454; font-weight:600; }
  .geo-c    { color:#ff6b6b; font-weight:600; }
  .dot      { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .dot-leo  { background:#4fd6c0; } .dot-meo { background:#ffb454; } .dot-geo { background:#ff6b6b; }
  .statline { font-size:13px; color:#8aa2b8; }
</style>
"""
scene.append_to_caption(CSS)

def section(title):
    scene.append_to_caption(f'<span class="sec-h">{title}</span><br>')

def slider_label(text, cls=""):
    scene.append_to_caption(f'<span class="lab {cls}">{text}: </span>')

def make_slider(text, mn, mx, val, step, on_change, cls=""):
    slider_label(text, cls)
    val_text = vp.wtext(text=f"{val}")
    def cb(s):
        val_text.text = f"{s.value:.0f}" if step >= 1 else f"{s.value:.2f}"
        on_change(s.value)
    scene.append_to_caption("<br>")
    vp.slider(min=mn, max=mx, value=val, step=step, bind=cb, length=280)
    scene.append_to_caption("<br>")
    return val_text

scene.append_to_caption(
    '<span class="title">SATELLITE CONSTELLATION COVERAGE SIMULATOR</span><br>'
    '<span class="subtitle">Drag empty space to orbit the camera &middot; scroll to zoom<br>'
    'Click a satellite, then drag it to move it along its own orbit ring</span><br>'
)

section('<span class="dot dot-leo"></span>LEO shell')
make_slider("Satellites", 0, 60, state["leo_n"], 1, lambda v: (state.update(leo_n=int(v)), generate_constellation()), "leo-c")
make_slider("Altitude (km)", 300, 2000, state["leo_alt"], 10, lambda v: (state.update(leo_alt=v), generate_constellation()), "leo-c")
make_slider("Inclination (deg)", 0, 98, state["leo_incl"], 1, lambda v: (state.update(leo_incl=v), generate_constellation()), "leo-c")

section('<span class="dot dot-meo"></span>MEO shell')
make_slider("Satellites", 0, 40, state["meo_n"], 1, lambda v: (state.update(meo_n=int(v)), generate_constellation()), "meo-c")
make_slider("Altitude (km)", 2000, 25000, state["meo_alt"], 100, lambda v: (state.update(meo_alt=v), generate_constellation()), "meo-c")
make_slider("Inclination (deg)", 0, 90, state["meo_incl"], 1, lambda v: (state.update(meo_incl=v), generate_constellation()), "meo-c")

section('<span class="dot dot-geo"></span>GEO ring')
make_slider("Satellites", 0, 30, state["geo_n"], 1, lambda v: (state.update(geo_n=int(v)), generate_constellation()), "geo-c")

section("Coverage model")
make_slider("Min elevation (deg)", 0, 40, state["min_el"], 1, lambda v: (state.update(min_el=v), generate_constellation()))
make_slider("Time scale (10^x real-time)", 0, 4, state["time_scale_exp"], 0.1, lambda v: state.update(time_scale_exp=v))

section("Playback")
scene.append_to_caption('<span class="lab">Status: </span>')
play_text = vp.wtext(text="Playing")
def toggle_play(b):
    state["playing"] = not state["playing"]
    play_text.text = "Playing" if state["playing"] else "Paused"
scene.append_to_caption("<br>")
vp.button(text="Play / Pause", bind=toggle_play)
scene.append_to_caption("&nbsp;&nbsp;")
vp.button(text="Regenerate", bind=lambda b: generate_constellation())
scene.append_to_caption("<br>")

section("Live stats")
scene.append_to_caption('<span class="statline">Coverage: </span>')
coverage_text = vp.wtext(text="0.0%  (0.00B / 7.90B people)")
scene.append_to_caption('<br><span class="statline">Selected: </span>')
sel_text = vp.wtext(text="none")
scene.append_to_caption("<br>")


# ============================== MOUSE INTERACTION =============================
def on_mousedown():
    obj = scene.mouse.pick
    if obj is not None and hasattr(obj, "sat_ref"):
        selected["sat"] = obj.sat_ref
        selected["dragging"] = True
        sel_text.text = f"Selected satellite: {obj.sat_ref.type} @ {obj.sat_ref.alt_km:.0f} km"

def on_mousemove():
    sat = selected["sat"]
    if selected["dragging"] and sat is not None:
        normal = vp.cross(sat.bx, sat.by)
        pt = scene.mouse.project(normal=normal, point=vp.vector(0, 0, 0))
        if pt is not None:
            local_x = vp.dot(pt, sat.bx)
            local_y = vp.dot(pt, sat.by)
            sat.phase_override = math.atan2(local_y, local_x)

def on_mouseup():
    sat = selected["sat"]
    if sat is not None and sat.phase_override is not None:
        sat.m0 = sat.phase_override - (2 * math.pi / sat.period) * sim_time_s
        sat.phase_override = None
    selected["dragging"] = False

scene.bind('mousedown', on_mousedown)
scene.bind('mousemove', on_mousemove)
scene.bind('mouseup', on_mouseup)


# ================================ MAIN LOOP ===================================
generate_constellation()
compute_coverage()

coverage_accum = 0.0
FPS = 60
dt_real = 1.0 / FPS
prev_earth_rot_deg = 0.0

while True:
    vp.rate(FPS)

    mult = 10 ** state["time_scale_exp"]
    if state["playing"]:
        sim_time_s += dt_real * mult
        earth_rotation_deg = (sim_time_s / SIDEREAL_DAY_S * 360) % 360

    # rotate earth incrementally (vpython has no absolute-rotation setter)
    delta_deg = earth_rotation_deg - prev_earth_rot_deg
    if delta_deg != 0:
        earth.rotate(angle=math.radians(delta_deg), axis=vp.vector(0, 1, 0), origin=vp.vector(0, 0, 0))
    prev_earth_rot_deg = earth_rotation_deg

    for sat, mesh, cone in sat_objs:
        sat.update(sim_time_s, earth_rotation_deg)
        mesh.pos = sat.pos
        if cone:
            direction = sat.pos.norm()
            half_ang = math.radians(sat.coverage_half_angle_deg(state["min_el"]))
            height = sat.pos.mag - SCENE_R * 0.98
            foot_r = max(0.01, sat.pos.mag * math.tan(half_ang) * 0.55)
            cone.pos = direction * (SCENE_R * 0.98)
            cone.axis = direction * height
            cone.radius = foot_r

    coverage_accum += dt_real
    if coverage_accum > 0.5:
        coverage_accum = 0.0
        compute_coverage()
        covered_people = coverage_stats["covered_millions"] / 1000.0
        total_people = TOTAL_POP_MILLIONS / 1000.0
        coverage_text.text = (f"Coverage: {coverage_stats['pct']:.1f}%  "
                               f"({covered_people:.2f}B / {total_people:.2f}B people)  |  "
                               f"Satellites: {len(satellites)}  |  "
                               f"Sim time: {sim_time_s/3600:.1f}h")
