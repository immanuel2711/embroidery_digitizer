import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from pyembroidery import EmbPattern, write_dst, STITCH, TRIM
import matplotlib.pyplot as plt

# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(image_path, width=512):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found!")
    aspect = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (width, int(width * aspect)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# -----------------------------
# Simplify image using KMeans
# -----------------------------
def simplify_colors(img_rgb, n_colors=6):
    h, w = img_rgb.shape[:2]
    flat = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(flat)
    clustered = kmeans.cluster_centers_.astype("uint8")[labels]
    clustered_img = clustered.reshape((h, w, 3))
    return clustered_img, labels.reshape((h, w)), kmeans.cluster_centers_

# -----------------------------
# Extract binary masks for each color
# -----------------------------
def extract_region_masks(labels, n_colors):
    masks = []
    for i in range(n_colors):
        mask = np.uint8(labels == i) * 255
        masks.append(mask)
    return masks

# -----------------------------
# Skeletonize edges
# -----------------------------
def extract_single_line_edges(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

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

# -----------------------------
# Smooth contours
# -----------------------------
def smooth_contour(contour, sigma=1.2):
    contour = contour[:, 0, :]
    if len(contour) < 5:
        return contour
    x = gaussian_filter1d(contour[:, 0], sigma)
    y = gaussian_filter1d(contour[:, 1], sigma)
    return np.stack((x, y), axis=1).astype(np.int32)

# -----------------------------
# Get all smooth contours from edge image
# -----------------------------
def extract_all_smooth_contours(edge_img, sigma=1.2, min_len=25):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    smoothed = []
    for cnt in contours:
        if len(cnt) >= min_len:
            smoothed.append(smooth_contour(cnt, sigma))
    return smoothed

# -----------------------------
# Add stitches from contour
# -----------------------------
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
    pattern.add_stitch_absolute(STITCH, x0, y0)

# -----------------------------
# Fill region with scanlines
# -----------------------------
def fill_area_with_scanlines(pattern, mask, scale=10, step=3):
    h, w = mask.shape
    for y in range(0, h, step):
        inside = False
        for x in range(w):
            if mask[y, x] > 0:
                if not inside:
                    pattern.add_stitch_absolute(STITCH, int(x * scale), int(y * scale))
                    inside = True
            else:
                if inside:
                    pattern.add_stitch_absolute(STITCH, int(x * scale), int(y * scale))
                    inside = False

# -----------------------------
# Create final DST pattern
# -----------------------------
def generate_combined_dst(contours, masks, centers, output_path="final_embroidery.dst", scale=10):
    pattern = EmbPattern()

    # First: add fill regions from color masks
    for idx, mask in enumerate(masks):
        color = centers[idx]
        pattern.add_thread({
            "red": int(color[0]),
            "green": int(color[1]),
            "blue": int(color[2]),
            "description": f"Fill {idx}",
            "catalog_number": str(idx)
        })

        # Fill region
        fill_area_with_scanlines(pattern, mask, scale=scale, step=3)
        pattern.add_command(TRIM)

    # Then: add black outline for contours
    pattern.add_thread({
        "red": 0, "green": 0, "blue": 0,
        "description": "Outline", "catalog_number": "000"
    })
    for cnt in contours:
        add_stitch_from_contour(pattern, cnt, scale=scale)
        pattern.add_command(TRIM)

    pattern.end()
    write_dst(pattern, output_path)
    print(f"✅ DST file with fills and outlines saved to: {output_path}")

# -----------------------------
# Visualization helper
# -----------------------------
def visualize_contours(image_rgb, contours):
    canvas = np.ones_like(image_rgb) * 255
    for cnt in contours:
        cv2.polylines(canvas, [cnt], isClosed=True, color=(0, 0, 0), thickness=1)
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Final Clean Contours")
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    image_path = "images/sample.jpg"  # <-- Change this to your image
    n_colors = 6
    scale = 10

    # Preprocess image
    img_rgb = preprocess_image(image_path)

    # Step 1: Extract region masks from clustered colors
    clustered_img, labels, centers = simplify_colors(img_rgb, n_colors=n_colors)
    masks = extract_region_masks(labels, n_colors=n_colors)

    # Optional: show masks
    plt.figure(figsize=(15, 5))
    for i, mask in enumerate(masks):
        plt.subplot(1, n_colors, i + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Color {i}")
        plt.axis("off")
    plt.show()

    # Step 2: Extract single-line contours for outline
    edge_single_line = extract_single_line_edges(img_rgb)
    contours = extract_all_smooth_contours(edge_single_line, sigma=1.2, min_len=25)

    # Optional: visualize contours
    visualize_contours(img_rgb, contours)

    # Step 3: Generate embroidery pattern with fills and outlines
    generate_combined_dst(contours, masks, centers, output_path="final_embroidery.dst", scale=scale)
