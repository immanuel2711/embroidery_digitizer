import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pyembroidery import EmbPattern, write_dst, STITCH
import matplotlib.pyplot as plt

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(image_path, width=512):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found!")
    aspect = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (width, int(width * aspect)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# -------------------------------
# Use Morphological Thinning (Skeletonization)
# -------------------------------
def extract_single_line_edges(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological thinning to reduce double edges
    skel = np.zeros(edges.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(edges, open_img)
        eroded = cv2.erode(edges, element)
        skel = cv2.bitwise_or(skel, temp)
        edges = eroded.copy()
        if cv2.countNonZero(edges) == 0:
            break

    return skel

# -------------------------------
# Smooth Contour
# -------------------------------
def smooth_contour(contour, sigma=1.2):
    contour = contour[:, 0, :]  # (N, 1, 2) -> (N, 2)
    if len(contour) < 5:
        return contour
    x = gaussian_filter1d(contour[:, 0], sigma)
    y = gaussian_filter1d(contour[:, 1], sigma)
    return np.stack((x, y), axis=1).astype(np.int32)

# -------------------------------
# Extract Internal + External Smooth Contours (No double lines)
# -------------------------------
def extract_all_smooth_contours(edge_img, sigma=1.2, min_len=25):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    smoothed = []
    for cnt in contours:
        if len(cnt) >= min_len:
            smoothed.append(smooth_contour(cnt, sigma))
    return smoothed

# -------------------------------
# Add Contour as Stitches (No black JUMPs)
# -------------------------------
def add_stitch_from_contour(pattern, contour, scale=10, min_dist=3):
    if len(contour) < 2:
        return
    x0, y0 = contour[0]
    x0, y0 = int(x0 * scale), int(y0 * scale)
    pattern.add_stitch_absolute(STITCH, x0, y0)
    prev = (x0, y0)

    for x, y in contour[1:]:
        x_scaled, y_scaled = int(x * scale), int(y * scale)
        dx, dy = x_scaled - prev[0], y_scaled - prev[1]
        if (dx ** 2 + dy ** 2) ** 0.5 >= min_dist:
            pattern.add_stitch_absolute(STITCH, x_scaled, y_scaled)
            prev = (x_scaled, y_scaled)

    pattern.add_stitch_absolute(STITCH, x0, y0)  # Close loop

# -------------------------------
# Generate DST
# -------------------------------
def generate_clean_detailed_dst(contours, output_path="peacock_clean_singleline.dst", scale=10):
    pattern = EmbPattern()
    pattern.add_thread({
        "red": 0, "green": 0, "blue": 0,
        "description": "Outline", "catalog_number": "000"
    })

    for cnt in contours:
        add_stitch_from_contour(pattern, cnt, scale=scale)

    pattern.end()
    write_dst(pattern, output_path)
    print(f"✅ Final single-line DST saved to: {output_path}")

# -------------------------------
# Visualize Result
# -------------------------------
def visualize(image_rgb, contours):
    canvas = np.ones_like(image_rgb) * 255
    for cnt in contours:
        cv2.polylines(canvas, [cnt], isClosed=True, color=(0, 0, 0), thickness=1)
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Final Clean Single-Line Contours")
    plt.show()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    image_path = "images/dore.webp"  # Change if needed
    img_rgb = preprocess_image(image_path)
    edge_single_line = extract_single_line_edges(img_rgb)
    contours = extract_all_smooth_contours(edge_single_line, sigma=1.2, min_len=25)

    visualize(img_rgb, contours)
    generate_clean_detailed_dst(contours, output_path="peacock_clean_singleline.dst", scale=10)
