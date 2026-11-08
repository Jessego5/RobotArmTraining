#!/usr/bin/env python3
"""The real (machined) planet carrier, measured from the October 30-31 wrist videos.

Zeb's Onshape carrier (sim/actuator, 04_Quad_planetary_carrier) is the plain design:
two 53 mm plates, four posts, an open 22.9 mm bore. The carrier in the recordings is
a machined version of it: smaller, with a ball bearing in the bore, raised bosses
around the post screws and four small holes. This builds that part from measurements,
in the CAD carrier's frame (origin at the centre of its bottom face, z up), so the gear
and pin keep their assembled places.

How it was measured: 120 right-wrist frames (15 demonstrations, carrier held face-on
to the camera) mapped onto the carrier's face plane through the fitted fisheye and
carrier pose, aligned and stacked into a 0.1 mm/pixel face-on photo
(--photo writes it). The camera cannot judge depth, so the photo's scale comes from
the planet holes, which the gears fix at 17.96 mm from the centre (factor 1.095).
Two checks agree with that scale: the bearing's outer edge measures 22 mm (a 608
bearing, 8 x 22 x 7 mm, in the CAD's 22.9 mm bore), and the right gripper, at its
recorded opening, holds the carrier with its pads 42.8 mm apart, plus the 3.6 mm the
fin-ray fingers give on the 23.95 mm gear: about 46-49 mm across, against the
photo's 48. The CAD's 53 mm would need 10 mm of give.

Measured (scaled): outer diameter 48 mm (+-1.5, the edge is the least sharp feature),
post screws at 21.2 mm on the diagonals (+-1: the sockets'
shadows blur them; the planet gear clears the posts by 0.5 mm), small holes 2.6 mm at 12.5 mm radius, 16
degrees before each planet hole. Kept from the CAD, which the camera cannot check:
the plate heights (bottom 0-3 mm, top 14.2-17.2 mm, bosses to 19.2 mm), the planet
holes (8.2 mm, which the 8 mm pin needs), the post diameter and the bottom plate's
35.5 mm opening. The bearing is modelled only within the top plate's thickness
(its real depth is unknown; 7 mm would reach the planet gear's layer). The curved
pockets machined into the face are not modelled.

    python tools/make_real_carrier.py      # -> sim/actuator/meshes/large_carrier_real/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import trimesh
from shapely import affinity, constrained_delaunay_triangles
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from tools.prepare_actuator import merge_convex

OUT = ROOT / 'sim/actuator/meshes/large_carrier_real'
MM = 1e-3
DIMS = dict(
    outer_d=48.2, planet_r=17.96, planet_hole_d=8.2, post_r=21.2, post_d=5.8, post_hole_d=3.0,
    boss_w=6.8, bottom_open_d=35.5, bearing_od=22.0, bearing_id=8.3, small_r=12.5, small_d=2.6,
    small_lead_deg=16., bottom=(0., 3.), posts=(3., 14.2), top=(14.2, 17.2), bosses=(17.2, 19.2),
)
SEGS = 24  # quarter-circle segments


def disc(d, at=(0., 0.)):
    return Point(*at).buffer(d / 2, quad_segs=SEGS)


def polar(r, deg):
    return r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg))


def layers(p=DIMS):
    """{name: (shapely outline in mm, z low, z high)}; the bearing is its own visual."""
    posts = [polar(p['post_r'], a) for a in (45, 135, 225, 315)]
    screw = unary_union([disc(p['post_hole_d'], c) for c in posts])
    plate = unary_union([disc(p['outer_d'])] + [disc(p['post_d'], c) for c in posts])
    planet = unary_union([disc(p['planet_hole_d'], polar(p['planet_r'], a)) for a in (0, 90, 180, 270)])
    small = unary_union([disc(p['small_d'], polar(p['small_r'], a - p['small_lead_deg'])) for a in (0, 90, 180, 270)])
    # A U-shaped boss around each post screw: the post's circle, run out radially past the rim.
    boss = []
    for a in (45, 135, 225, 315):
        run = box(0, -p['boss_w'] / 2, p['outer_d'] / 2 + 1 - p['post_r'], p['boss_w'] / 2)
        run = affinity.rotate(affinity.translate(run, p['post_r']), a, origin=(0, 0))
        boss.append(unary_union([disc(p['boss_w'], polar(p['post_r'], a)), run]))
    boss = unary_union(boss).intersection(disc(p['outer_d'] + .2)).difference(screw)
    return {
        'bottom': (plate.difference(disc(p['bottom_open_d'])).difference(screw), *p['bottom']),
        'posts': (unary_union([disc(p['post_d'], c) for c in posts]).difference(screw), *p['posts']),
        'top': (plate.difference(disc(p['bearing_od'])).difference(planet).difference(small).difference(screw), *p['top']),
        'bosses': (boss, *p['bosses']),
    }


def bearing(p=DIMS):
    """608-style bearing faces, flush with the top face: inner race, recessed shield, outer race."""
    top = p['top'][1]
    rings = [(p['bearing_id'], 12.0, top - .1), (12.0, 19.4, top - .35), (19.4, p['bearing_od'], top - .1)]
    return [(disc(o).difference(disc(i)), p['top'][0], z) for i, o, z in rings]


def extrude(shape, low, high):
    polys = list(shape.geoms) if hasattr(shape, 'geoms') else [shape]
    parts = []
    for poly in polys:
        m = trimesh.creation.extrude_polygon(poly, (high - low) * MM)
        m.apply_scale([MM, MM, 1])
        m.apply_translation([0, 0, low * MM])
        parts.append(m)
    return parts


def prisms(shape, low, high):
    """Convex prisms covering an outline (as tools/prepare_actuator.py does for the CAD parts)."""
    polys = list(shape.geoms) if hasattr(shape, 'geoms') else [shape]
    out = []
    for poly in polys:
        poly = poly.simplify(.03, preserve_topology=True)  # mm: fewer slivers along the circles
        poly = Polygon(np.asarray(poly.exterior.coords) * MM, [np.asarray(r.coords) * MM for r in poly.interiors])
        for piece in merge_convex(list(constrained_delaunay_triangles(poly).geoms)):
            pts = np.asarray(piece.exterior.coords)[:-1]
            out.append(trimesh.convex.convex_hull(np.vstack([np.c_[pts, np.full(len(pts), z * MM)] for z in (low, high)])))
    return out


def build(out=OUT, p=DIMS):
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('collision_*.stl'):
        old.unlink()
    body = layers(p)
    visual = trimesh.util.concatenate([m for shape, lo, hi in body.values() for m in extrude(shape, lo, hi)])
    visual.export(out / 'visual.stl')
    rings = trimesh.util.concatenate([m for shape, lo, hi in bearing(p) for m in extrude(shape, lo, hi)])
    rings.export(out / 'bearing.stl')
    # Collision: the body's layers, and the bearing as a solid ring (its faces are only visual).
    pieces = [c for shape, lo, hi in body.values() for c in prisms(shape, lo, hi)]
    pieces += prisms(disc(p['bearing_od']).difference(disc(p['bearing_id'])), *p['top'])
    for k, piece in enumerate(pieces):
        piece.export(out / f'collision_{k:03d}.stl')
    volume = sum(c.volume for c in pieces)
    centre = sum(c.volume * c.center_mass for c in pieces) / volume
    info = dict(dims_mm=p, collision_pieces=len(pieces), volume_m3=volume, mass_kg=volume * 2700,
                centre_of_mass=centre.tolist(), bounds=visual.bounds.tolist())
    (out / 'carrier.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    info = build(args.output)
    print(f"wrote {args.output}: {info['collision_pieces']} collision pieces, {info['mass_kg'] * 1000:.0f} g of aluminium, "
          f"bounds {np.round(np.array(info['bounds']) * 1000, 1).tolist()} mm")


if __name__ == '__main__':
    main()
