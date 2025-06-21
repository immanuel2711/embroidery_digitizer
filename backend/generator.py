import cv2
import numpy as np
import os
from pyembroidery import EmbPattern, STITCH, JUMP, write_dst

def smooth_contour(contour, epsilon_ratio=0.001):
    """Smooth the contour while preserving curves."""
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def generate_dst_from_image(image_path, dst_output_path, resize_to=(300, 300), min_area=80):
    # 1. Load and resize
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image.")
        return False

    print("📂 Image loaded successfully.")

    img = cv2.resize(img, resize_to)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive thresholding for universal edge detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # 4. Denoising and edge refinement
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary = cv2.medianBlur(binary, 3)

    # 5. Find all contours (including holes)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 5.1 Optional: sort contours by proximity to reduce stitch travel
    def contour_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    contours = sorted(contours, key=lambda c: contour_centroid(c))

    print(f"✂️ Total valid contours: {len(contours)}")

    pattern = EmbPattern()
    current_x, current_y = 0, 0

    for contour in contours:
        # 6. Smooth contour
        contour = smooth_contour(contour, epsilon_ratio=0.001)
        points = contour.squeeze()

        if len(points.shape) != 2 or len(points) < 5:
            continue

        # 7. Begin stitching with a jump only if far from previous
        start = points[0]
        start_x, start_y = int(start[0]), int(start[1])

        # Distance threshold to decide jump
        distance = np.linalg.norm(np.array([current_x, current_y]) - np.array([start_x, start_y]))
        if distance > 10:  # threshold in pixels
            pattern.add_command(2)  # STOP (breaks long jump line in viewer)
            pattern.add_stitch_absolute(JUMP, start_x, start_y)
        pattern.add_stitch_absolute(STITCH, start_x, start_y)

        for pt in points[1:]:
            x, y = int(pt[0]), int(pt[1])
            pattern.add_stitch_absolute(STITCH, x, y)

        # Update current needle position
        current_x, current_y = x, y

    # 8. End pattern
    pattern.end()

    os.makedirs(os.path.dirname(dst_output_path), exist_ok=True)
    write_dst(pattern, dst_output_path)
    print(f"✅ Top-quality DST saved: {dst_output_path}")
    return True
