import cv2
import numpy as np
import os
from pyembroidery import EmbPattern, STITCH, JUMP, write_dst

def smooth_contour(contour, epsilon_ratio=0.0001):  # VERY low = preserve curves better
    """Smooth the contour while preserving curves."""
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def generate_dst_from_image(image_path, dst_output_path, resize_to=(300, 300), min_area=50):
    # 1. Load and resize
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image.")
        return False

    print("📂 Image loaded successfully.")
    img = cv2.resize(img, resize_to)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive thresholding (detect faint edges too)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2  # smaller constant = catch lighter lines
    )

    # 4. Denoising and enhancement
    binary = cv2.medianBlur(binary, 3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Optional: enhance thin lines
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    # 5. Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours by proximity (to avoid messy stitch travel)
    def contour_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    contours = sorted(contours, key=lambda c: contour_centroid(c))

    print(f"✂️ Total high-quality contours: {len(contours)}")

    pattern = EmbPattern()
    current_x, current_y = 0, 0

    for contour in contours:
        # 6. Smooth but preserve detail
        contour = smooth_contour(contour, epsilon_ratio=0.0001)
        points = contour.squeeze()

        if len(points.shape) != 2 or len(points) < 5:
            continue

        # Densify points if shape is long
        new_points = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            dist = np.linalg.norm(p2 - p1)
            if dist > 2:
                interp_count = int(dist // 1.5)
                for j in range(interp_count):
                    ratio = j / interp_count
                    interp_point = (1 - ratio) * p1 + ratio * p2
                    new_points.append(interp_point)
            new_points.append(p2)
        points = np.array([points[0]] + new_points, dtype=np.int32)

        # 7. Jump only if far from last point
        start_x, start_y = points[0]
        distance = np.linalg.norm(np.array([current_x, current_y]) - np.array([start_x, start_y]))

        if distance > 5:
            pattern.add_command(2)  # STOP
            pattern.add_stitch_absolute(JUMP, int(start_x), int(start_y))
        pattern.add_stitch_absolute(STITCH, int(start_x), int(start_y))

        for pt in points[1:]:
            x, y = int(pt[0]), int(pt[1])
            pattern.add_stitch_absolute(STITCH, x, y)

        current_x, current_y = x, y

    # 8. Finalize pattern
    pattern.end()

    os.makedirs(os.path.dirname(dst_output_path), exist_ok=True)
    write_dst(pattern, dst_output_path)
    print(f"🏁 DST file ready for machines: {dst_output_path}")
    return True
