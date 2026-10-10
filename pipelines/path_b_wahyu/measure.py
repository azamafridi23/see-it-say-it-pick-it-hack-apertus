"""Extract foreground proposals and safe interior pixels with OpenCV.

Proposals are geometric evidence, not confirmed semantic objects. Apertus assigns
identities later; contours may still merge touching objects or split one object.
"""

import json

import cv2
import numpy as np


def detect_objects(
    image,
    empty_image=None,
    min_area=5000,
    max_area_ratio=0.12,
    saturation_threshold=85,
    brightness_threshold=140,
):
    """Return candidate metadata and a binary foreground mask in original pixels.

    An optional empty-workcell image enables LAB background subtraction. Without
    it, separate color masks avoid joining yellow objects to touching red ones.
    Thresholds are tuned for the challenge's fixed camera and workcell lighting.
    """
    if image is None:
        raise ValueError("Invalid input image")
    height, width = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    if empty_image is not None:
        if empty_image.shape != image.shape:
            raise ValueError("Empty image must have the same dimensions")
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        background = cv2.cvtColor(
            cv2.GaussianBlur(empty_image, (5, 5), 0), cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        difference = np.linalg.norm(lab - background, axis=2)
        mask = (difference > 28).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Segment colors separately; closing their union would merge stacked gears.
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        colored = (sat >= saturation_threshold) & (val >= 65)
        red = colored & ((hue < 12) | (hue >= 170))
        yellow = colored & (hue >= 12) & (hue < 45)
        other = colored & ~(red | yellow)
        neutral = (sat < saturation_threshold) & (val >= brightness_threshold)
        region_masks = [m.astype(np.uint8) * 255 for m in (red, yellow, other, neutral)]
        region_masks = [
            cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open) for m in region_masks
        ]
        contours = []
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in region_masks:
            region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.bitwise_or(mask, region)
            found, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(found)

    objects = []

    image_area = height * width

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noise
        if area < min_area:
            continue

        # Ignore very large regions
        if area > max_area_ratio * image_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small shapes
        if w < 20 or h < 20:
            continue

        # Ignore regions occupying most of the image
        if w > width * 0.55:
            continue

        if h > height * 0.65:
            continue

        # Filter long thin structures
        aspect_ratio = max(w / h, h / w)

        if aspect_ratio > 5:
            continue

        # Ignore regions touching image boundaries
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        # Choose a point on foreground, away from holes and contour boundaries.
        local = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([[[x, y]]])
        cv2.drawContours(local, [shifted], -1, 255, -1)
        local = cv2.bitwise_and(local, mask[y : y + h, x : x + w])
        distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
        _, _, _, point = cv2.minMaxLoc(distance)
        cx, cy = x + point[0], y + point[1]

        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, [shifted], -1, 255, -1)
        hsv_crop = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
        samples = hsv_crop[filled > 0]
        colored_samples = samples[samples[:, 1] >= saturation_threshold]
        color = "neutral"
        if len(colored_samples) > len(samples) * 0.35:
            hue_median = float(np.median(colored_samples[:, 0]))
            color = (
                "red"
                if hue_median < 12 or hue_median >= 170
                else (
                    "orange"
                    if hue_median < 24
                    else (
                        "yellow"
                        if hue_median < 45
                        else "green" if hue_median < 90 else "blue"
                    )
                )
            )
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        perimeter = cv2.arcLength(contour, True)
        solidity = area / hull_area if hull_area else 0
        circularity = 4 * np.pi * area / perimeter**2 if perimeter else 0
        vertices = len(cv2.approxPolyDP(contour, 0.025 * perimeter, True))
        hint = "unknown"
        if color == "red":
            hint = "red_gear" if solidity < 0.88 or vertices > 6 else "red_cube"
        elif color in ("yellow", "orange"):
            hint = (
                "yellow_gear"
                if solidity < 0.83 and vertices > 6
                else "banana" if aspect_ratio > 1.6 else "mandarin_or_dish"
            )
        objects.append(
            {
                "id": "",
                "pixel": [cx, cy],
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area_px": int(area),
                "color": color,
                "solidity": round(solidity, 3),
                "circularity": round(circularity, 3),
                "aspect_ratio": round(aspect_ratio, 3),
                "polygon_vertices": vertices,
                "category_hint": hint,
            }
        )

    # Stable reading order gives reproducible candidate IDs within a scene.

    objects.sort(key=lambda obj: (obj["pixel"][1], obj["pixel"][0]))

    for index, obj in enumerate(objects, 1):
        obj["id"] = f"obj_{index}"

    return objects, mask


def show_detections(image, objects, save_path=None):
    """
    Visualize object bounding boxes and pixel centers.
    """

    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.imshow(rgb)

    for obj in objects:

        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["pixel"]

        # Bounding box
        rectangle = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2
        )

        ax.add_patch(rectangle)

        # Center point
        ax.scatter(cx, cy, color="red", marker="x", s=40)

        # Object label
        ax.text(
            x1,
            max(0, y1 - 10),
            f'{obj["id"]} ({cx}, {cy})',
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 3},
        )

    ax.set_title(f"Detected Objects: {len(objects)}")

    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def show_mask(mask):
    """
    Visualize the binary mask used for detection.
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))

    plt.imshow(mask, cmap="gray")

    plt.title("Foreground Segmentation Mask")
    plt.axis("off")
    plt.show()


def main():
    """Run the original interactive detector preview for scene_01."""

    image = cv2.imread("./dataset/scene_01/rgb.png")

    if image is None:
        raise FileNotFoundError("rgb.png not found")

    # Optional background reference
    # Set to cv2.imread("empty_trays.png") if available
    empty_image = None

    # Detect all object candidates
    objects, mask = detect_objects(
        image=image,
        empty_image=empty_image,
        min_area=5000,
        max_area_ratio=0.12,
        saturation_threshold=85,
        brightness_threshold=140,
    )

    # Print JSON
    print(json.dumps(objects, indent=4))

    # Save JSON
    with open("detections.json", "w") as f:
        json.dump(objects, f, indent=4)

    # Show object detections
    show_detections(image, objects, save_path="detections.png")

    # Show segmentation mask
    show_mask(mask)


if __name__ == "__main__":
    main()
