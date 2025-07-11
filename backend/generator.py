import cv2
import numpy as np
import os
from pyembroidery import EmbPattern, STITCH, JUMP, write_dst, EmbThread

def preprocess_image(img, resize_to=(400, 400)):
    img = cv2.resize(img, resize_to)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Strong binarization using Otsu's method
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Only open (no close) to avoid filling thin gaps
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned

def interpolate_points(points, max_stitch_length=4.0):
    new_points = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        dist = np.linalg.norm(p2 - p1)
        interp_count = max(int(dist // max_stitch_length), 1)
        for j in range(interp_count):
            ratio = j / interp_count
            interp = (1 - ratio) * p1 + ratio * p2
            new_points.append(interp)
    new_points.append(points[-1])  # Ensure final point included
    return np.array(new_points, dtype=np.int32)

def generate_dst_from_image(image_path, dst_output_path, preview_path=None,
                            resize_to=(400, 400), min_area=5,
                            max_stitch_length=4.0, max_stitch_count=50000):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image failed to load.")
        print("📂 Image loaded successfully.")

        binary_mask = preprocess_image(img, resize_to=resize_to)

        pattern = EmbPattern()
        current_x, current_y = 0, 0
        total_stitches = 0
        preview = np.full((resize_to[1], resize_to[0], 3), 255, dtype=np.uint8)

        # Add thread
        thread = EmbThread()
        thread.set_color(0, 0, 0)
        thread.description = "Outline"
        pattern.add_thread(thread)

        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        if hierarchy is None:
            print("⚠️ No contours found.")
            return False

        hierarchy = hierarchy[0]
        print(f"🔍 Found {len(contours)} contours with hierarchy.")

        for idx, (contour, hier) in enumerate(zip(contours, hierarchy)):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area < min_area or perimeter < 20:
                continue

            # Keep narrow but tall child contours (gaps between letters)
            parent = hier[3]
            if parent != -1:
                parent_area = cv2.contourArea(contours[parent])
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / float(w + 1e-5)

                if area < 0.01 * parent_area and aspect_ratio < 2.5:
                    continue  # likely noise

            points = contour.squeeze()
            if len(points.shape) != 2 or len(points) < 5:
                continue

            points = interpolate_points(points, max_stitch_length)
            start_x, start_y = points[0]

            if np.linalg.norm([current_x - start_x, current_y - start_y]) > 2:
                pattern.add_command(2)  # STOP
                pattern.add_stitch_absolute(JUMP, int(start_x), int(start_y))

            pattern.add_stitch_absolute(STITCH, int(start_x), int(start_y))
            total_stitches += 1

            for pt in points[1:]:
                if total_stitches >= max_stitch_count:
                    print("⚠️ Max stitch count reached.")
                    break
                x, y = int(pt[0]), int(pt[1])
                pattern.add_stitch_absolute(STITCH, x, y)
                cv2.circle(preview, (x, y), 0, (0, 0, 0), 1)
                total_stitches += 1

            current_x, current_y = x, y

        pattern.end()
        os.makedirs(os.path.dirname(dst_output_path), exist_ok=True)
        write_dst(pattern, dst_output_path)
        print(f"✅ DST file saved to: {dst_output_path}")

        if preview_path:
            cv2.imwrite(preview_path, preview)
            print(f"🖼️ Preview image saved to: {preview_path}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
