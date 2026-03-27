"""CadQuery 2.x reference cheat sheet — included verbatim in LLM system prompts."""

CADQUERY_CHEAT_SHEET = """\
# CadQuery 2.x Cheat Sheet

## *** UNIT SYSTEM: MILLIMETERS ***
# CadQuery / OCC uses MILLIMETERS internally.
# The STEP exporter writes coordinates in mm; the SimReady pipeline
# then detects the unit and scales by 0.001 to produce correct meter values.
#
# CONVERSION: blueprint dimensions are in meters → multiply by 1000 for CadQuery code.
#   0.24 m  →  240 mm   (write 240 in code)
#   0.05 m  →   50 mm   (write 50 in code)
#   0.003 m →    3 mm   (write 3 in code)
#
# Assign final shape to `result`. Do NOT call any export functions.

## Import & Entry Point
import cadquery as cq

## Workplane Construction
cq.Workplane("XY")          # Z-up horizontal plane (default — use this most)
cq.Workplane("XZ")          # Y-up vertical plane
cq.Workplane("YZ")          # X-up side plane

## Primitives (all centered at origin unless noted)
.box(length, width, height)               # centered=True by default
.cylinder(height, radius)                 # centered=True
.sphere(radius)
.cone(height, radius1, radius2=0)         # radius2=0 → sharp cone; radius2>0 → frustum

## Sketch → Solid
.rect(width, height)                      # 2D rectangle on current workplane
.circle(radius)                           # 2D circle
.ellipse(x_radius, y_radius)
.polygon(nSides, circumradius)
.polyline([(x0,y0),(x1,y1),...]).close()  # arbitrary 2D outline
.extrude(distance)                        # extrude sketch normal to workplane
.revolve(angleDeg, (ax,ay,az), (bx,by,bz))  # revolve around axis AB (both pts on sketch plane)

## Boolean Operations (on Workplane objects)
a.cut(b)        # a minus b — b must overlap a
a.union(b)      # merge a and b
a.intersect(b)  # keep only overlapping region

## Transforms (return new Workplane — capture the result)
.translate((x, y, z))                      # shift in world space (mm)
.rotate((ax,ay,az), (bx,by,bz), angleDeg)  # rotate degrees around axis A→B
.mirror("XY")                              # mirror across XY plane ("XZ" or "YZ" also valid)

## Face/Edge Selection (for workplane shifting or feature attachment)
.faces(">Z")   # face with max Z extent (top face for a box standing on Z)
.faces("<Z")   # face with min Z extent (bottom face)
.faces("#Z")   # faces whose normal is perpendicular to Z (side faces)
.faces("|Z")   # faces whose normal is parallel to Z
.edges(">Z"), .vertices(">Z")  # same selectors for edges/vertices

## Working on a Selected Face
.faces(">Z").workplane()  # shifts origin to center of selected face
.faces(">Z").workplane().hole(diameter)  # drill through hole from top
.faces(">Z").workplane().circle(r).extrude(h)  # add boss

## Common Feature Operations
.hole(diameter, depth=None)   # drill hole through (depth=None → all the way through)
.shell(thickness)             # hollow the solid (negative = inward)
.fillet(radius)               # round all edges (or select edges first)
.chamfer(length)              # bevel all edges

## Assembly
assy = cq.Assembly()
assy.add(part1, name="base",  loc=cq.Location(cq.Vector(0, 0, 0)))
assy.add(part2, name="top",   loc=cq.Location(cq.Vector(0, 0, 50)))   # 50 mm offset
# Rotation: cq.Location(cq.Vector(x,y,z), cq.Vector(ax,ay,az), angle_deg)
result = assy.toCompound()    # MUST convert to Compound for STEP export

## Typical Patterns (all dimensions in mm)

# Box with centered cylindrical hole  (10×10×5 cm → 100×100×50 mm)
result = (cq.Workplane("XY")
    .box(100, 100, 50)
    .faces(">Z").workplane()
    .hole(30))                  # 30 mm diameter hole

# Flanged pipe (revolve cross-section)  (outer_r=50mm, inner_r=20mm, h=80mm)
result = (cq.Workplane("XZ")
    .polyline([(20,0),(50,0),(50,5),(25,5),(25,80),(20,80)])
    .close()
    .revolve(360, (0,0,0), (0,1,0)))

# L-bracket  (100×20×60 mm plate + 20×60×60 mm wing)
plate = cq.Workplane("XY").box(100, 20, 60)
wing  = cq.Workplane("XY").box(20, 60, 60).translate((40, 40, 0))
result = plate.union(wing)

## CRITICAL RULES
1. ALL dimensions in MILLIMETERS. Convert meters→mm by ×1000. (0.24 m = 240 mm)
2. Variable `result` must hold the final shape — executor exports it automatically.
3. Do NOT call cq.exporters.export() or any file I/O — the executor handles that.
4. Capture every .translate()/.rotate() — they return new objects: `a = a.translate(...)`.
5. Boolean ops require actual overlap; check that shapes intersect before .cut()/.intersect().
6. After .faces(">Z").workplane() the coordinate origin shifts — recompute offsets from there.
7. .hole(diameter) takes DIAMETER not radius (both in mm).
8. Assembly → STEP: always `result = assy.toCompound()` at the end.
9. Keep code self-contained — do not read external files or use random numbers.
"""
