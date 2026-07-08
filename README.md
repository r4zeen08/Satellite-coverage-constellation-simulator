# 🛰️ Satellite Constellation Coverage Simulator

An interactive 3D simulation exploring a simple but deep question: **how many people on Earth does a given satellite constellation actually cover — and how does that change as you add satellites, shift altitudes, or reshape orbital planes?**

The simulator models three constellation shells — **LEO, MEO, and GEO** — each independently configurable (satellite count, altitude, inclination). Orbits are propagated using real two-body orbital mechanics (Kepler's third law, `T = 2π√(a³/μ)`), not scripted animation, so satellite motion, periods, and coverage footprints all respond correctly to whatever parameters you set.

Each satellite projects a coverage cone derived from horizon geometry and a configurable minimum elevation angle — the same math used in real coverage/link-budget analysis. Coverage is checked against an approximate global population model (55 weighted regional hotspots blended into a lat/lon grid, normalized to ~7.9B people), giving a live **% of world population covered** stat as you reconfigure the constellation.

You can also click and drag individual satellites — motion is constrained to their own orbital plane, so you can re-phase a satellite without breaking its orbit's physics.

---

## Two implementations

| File | Runs on | Notes |
|---|---|---|
| `satellite_coverage_simulation.html` | Any modern browser | Three.js/WebGL. No install — just open the file. |
| `satellite_coverage_simulation.py` | Python 3 | VPython port with identical physics and population model. |

---

## Quick start — browser version

No installation needed.

1. Download `satellite_coverage_simulation.html`
2. Double-click it (or open it in Chrome/Firefox/Edge)
3. Done — it runs entirely client-side.

## Quick start — Python version

```bash
pip install vpython pillow numpy
python satellite_coverage_simulation.py
```

This opens a browser tab automatically — VPython renders its 3D view through a local web view, but all the physics and logic run in Python.

> If you hit `ModuleNotFoundError: No module named 'pkg_resources'`, install setuptools: `pip install setuptools`

---

## Controls

- **Orbit the camera** — click and drag empty space
- **Zoom** — scroll wheel
- **Move a satellite** — click it, then drag; it slides along its own orbital ring only, so its altitude and inclination stay physically valid
- **Sliders** — satellite count / altitude / inclination per shell (LEO, MEO, GEO), minimum elevation angle (controls coverage cone size), and simulation time scale
- **Regenerate** — redistributes satellites evenly across orbital planes using current slider values

---

## How it works

**Orbital mechanics** — each satellite is a circular Keplerian orbit defined by semi-major axis, inclination, RAAN (right ascension of ascending node), and initial phase. Position is propagated as `θ(t) = θ₀ + (2π/T)·t`, then rotated into 3D by inclination and RAAN. Period comes directly from Kepler's third law using Earth's real gravitational parameter (μ = 398,600 km³/s²).

**Coverage geometry** — a satellite's usable coverage cone depends on both altitude and a minimum elevation angle (how low toward the horizon a ground receiver can still see it). The half-angle of that cone is:

```
λ = arccos( (R⊕ / (R⊕ + h)) · cos(ε) ) − ε
```

where `ε` is the minimum elevation angle and `h` is altitude.

**Population model** — 55 hand-placed regional weights (lat, lon, population, spread) are blended as 2D gaussians onto a 180×90 lat/lon grid, plus a flat rural baseline, then normalized to ~7.9 billion total. This is a *hand-built approximation*, not census data — see "Limitations" below.

**Coverage %** — every grid cell's population is counted as "covered" if it falls within any satellite's coverage cone, based on angular distance from that satellite's sub-satellite point.

---

## Limitations / known simplifications

- Orbits are circular (eccentricity = 0) — no elliptical or perturbed orbits
- No atmospheric drag, J2 perturbation, or orbital decay
- Population distribution is an approximation built from ~55 regional weight points, not a real gridded census dataset
- Coverage counts a cell as "covered" by *any* satellite in range — no modeling of handoff, latency, or bandwidth limits

---

## Contributing

PRs welcome, especially:
- Swapping the population approximation for a real gridded dataset (e.g. [NASA SEDAC GPWv4](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) or WorldPop)
- Elliptical/perturbed orbit support
- Additional constellation presets (e.g. real-world configs like Starlink, GPS, Iridium shells)

## License

MIT — do whatever you want with it, just don't hold the author liable if you use this to plan an actual satellite constellation (please don't — go find real orbital mechanics software for that).
