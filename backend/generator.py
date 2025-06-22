import cv2
import numpy as np
import os
from pyembroidery import EmbPattern, STITCH, JUMP, write_dst, EmbThread

def smooth_contour(contour, epsilon_ratio=0.00002):
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def quantize_image(image, k=6):
    Z = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(image.shape)
    return quantized, labels.reshape(image.shape[:2]), centers

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
    return np.array([points[0]] + new_points, dtype=np.int32)

def generate_dst_from_image(image_path, dst_output_path, preview_path=None,
                            resize_to=(300, 300), min_area=5,
                            max_stitch_length=4.0, max_stitch_count=50000):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image failed to load.")
        print("📂 Image loaded successfully.")

        img = cv2.resize(img, resize_to)
        quantized, label_map, color_palette = quantize_image(img, k=6)

        pattern = EmbPattern()
        current_x, current_y = 0, 0
        total_stitches = 0
        preview = np.full((resize_to[1], resize_to[0], 3), 255, dtype=np.uint8)

        for color_index, color in enumerate(color_palette):
            print(f"🎨 Processing color {color_index + 1}/{len(color_palette)}... RGB: {tuple(color)}")

            thread = EmbThread()
            thread.set_color(color[2], color[1], color[0])  # BGR to RGB
            thread.description = f"Color-{color_index + 1}"
            thread.catalog_number = str(color_index + 1)
            pattern.add_thread(thread)

            mask = (label_map == color_index).astype(np.uint8) * 255
            mask = cv2.medianBlur(mask, 3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if hierarchy is None:
                continue

            def centroid(c):
                M = cv2.moments(c)
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

            valid = [(cnt, h) for cnt, h in zip(contours, hierarchy[0]) if cv2.contourArea(cnt) >= min_area]
            valid = sorted(valid, key=lambda ch: centroid(ch[0]))

            for contour, _ in valid:
                contour = smooth_contour(contour)
                points = contour.squeeze()
                if len(points.shape) != 2 or len(points) < 5:
                    continue

                points = interpolate_points(points, max_stitch_length)
                start_x, start_y = points[0]
                dist = np.linalg.norm(np.array([current_x, current_y]) - np.array([start_x, start_y]))
                if dist > 2:
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

                for pt in reversed(points):
                    if total_stitches >= max_stitch_count:
                        break
                    x, y = int(pt[0]), int(pt[1])
                    pattern.add_stitch_absolute(STITCH, x, y)
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
