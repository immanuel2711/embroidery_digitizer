import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path, width=300):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at '{image_path}' not found.")
    aspect = img.shape[0] / img.shape[1]
    resized_img = cv2.resize(img, (width, int(width * aspect)))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return img_rgb

def generate_tatami_fill(mask, spacing=1):
    height, width = mask.shape
    filled = np.zeros_like(mask)

    for y in range(0, height, spacing):
        in_line = False
        for x in range(width):
            if mask[y, x] == 255 and not in_line:
                start_x = x
                in_line = True
            elif (mask[y, x] != 255 or x == width - 1) and in_line:
                end_x = x if mask[y, x] != 255 else x + 1
                filled[y, start_x:end_x] = 255
                in_line = False

    return cv2.bitwise_and(filled, mask)

def apply_color_fill(filled_mask, color_rgb):
    color_img = np.zeros((filled_mask.shape[0], filled_mask.shape[1], 3), dtype=np.uint8)
    color_img[filled_mask == 255] = color_rgb
    return color_img

if __name__ == "__main__":
    image_path = "images/sample.jpg"  # ✅ Match Phase 1 image
    mask_folder = "masks"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load reference image (same size as masks)
    img_rgb = preprocess_image(image_path, width=300)
    h, w = img_rgb.shape[:2]
    final_output = np.ones((h, w, 3), dtype=np.uint8) * 255  # white canvas

    # Step 2: Load masks
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])
    for idx, filename in enumerate(mask_files):
        print(f"🪡 Processing region {idx} -> {filename}")
        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Step 3: Generate tatami fill
        tatami = generate_tatami_fill(mask, spacing=1)

        # Step 4: Get average color for thread
        masked_pixels = img_rgb[mask == 255]
        if masked_pixels.size == 0:
            continue
        avg_color = np.mean(masked_pixels, axis=0).astype(np.uint8)

        # Step 5: Apply fill color to tatami region
        filled_color = apply_color_fill(tatami, tuple(avg_color))
        mask_3ch = cv2.merge([mask] * 3)
        final_output = np.where(mask_3ch == 255, filled_color, final_output)

    # Step 6: Save and show result
    output_path = os.path.join(output_folder, "final_tatami_fill_realcolor.png")
    cv2.imwrite(output_path, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))

    plt.imshow(final_output)
    plt.title("🧵 Tatami Fill (Embroidery-style Avg Color)")
    plt.axis("off")
    plt.show()

    print(f"✅ Embroidery-style fill saved to: {output_path}")
