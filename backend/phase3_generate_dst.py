import json
import os
from pyembroidery import EmbPattern, STITCH, COLOR_CHANGE, write_dst, EmbThread

def add_tatami_fill(pattern, contour, step=2):
    """
    Fills a closed contour using horizontal tatami (zigzag) lines.
    `contour`: list of (x, y) tuples, already scaled to 0.1mm units
    """
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union

    poly = Polygon(contour)
    if not poly.is_valid or poly.area == 0:
        return

    min_y, max_y = int(poly.bounds[1]), int(poly.bounds[3])

    last_x, last_y = 0, 0
    for y in range(min_y, max_y, step):
        line = LineString([(poly.bounds[0], y), (poly.bounds[2], y)])
        intersection = poly.intersection(line)

        if intersection.is_empty:
            continue

        if intersection.geom_type == 'MultiLineString':
            segments = list(intersection.geoms)
        elif intersection.geom_type == 'LineString':
            segments = [intersection]
        else:
            continue

        for seg in segments:
            coords = list(seg.coords)
            if len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                pattern.add_stitch_absolute(STITCH, x1, y1)
                pattern.add_stitch_absolute(STITCH, x2, y2)
                last_x, last_y = x2, y2

def add_outline(pattern, contour):
    """
    Add a single outline run stitch along the contour.
    """
    first = True
    for x, y in contour:
        if first:
            pattern.add_stitch_absolute(STITCH, x, y)
            first = False
        else:
            pattern.add_stitch_absolute(STITCH, x, y)
    # Close loop
    pattern.add_stitch_absolute(STITCH, contour[0][0], contour[0][1])

def generate_dst_from_contours(json_dir="contours_json", centers_path="centers.pkl", output_path="output/embroidery.dst"):
    import pickle

    # ✅ Load color centers
    with open(centers_path, "rb") as f:
        centers = pickle.load(f)

    pattern = EmbPattern()

    files = sorted(os.listdir(json_dir))
    for idx, file in enumerate(files):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(json_dir, file), "r") as f:
            contours = json.load(f)

        # Convert center to RGB
        color = centers[idx].astype(int)
        r, g, b = int(color[0]), int(color[1]), int(color[2])

        # ✅ Skip black regions
        if (r, g, b) == (0, 0, 0):
            print(f"⏭️ Skipping black region ({file})")
            continue

        # ✅ Add thread color
        thread = EmbThread()
        thread.set_color(r, g, b)
        thread.description=f"Region {idx}"
        pattern.add_thread(thread)

        for contour in contours:
            points = [pt[0] for pt in contour]  # From [[[x, y]], [[x, y]], ...]
            add_tatami_fill(pattern, points)
            add_outline(pattern, points)

        pattern.add_command(COLOR_CHANGE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_dst(pattern, output_path)
    print(f"✅ DST file written to {output_path}")

if __name__ == "__main__":
    generate_dst_from_contours()
