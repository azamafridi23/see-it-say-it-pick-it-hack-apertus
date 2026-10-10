import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def detect_objects(
    image,
    empty_image=None,
    min_area=1200,
    max_area_ratio=0.12,
    saturation_threshold=85,
    brightness_threshold=175
):
    """
    Detect object candidates across the entire image.

    No tray ROI or tray detection required.

    Returns:
        objects: list of detected object candidates
        mask: binary foreground mask
    """

    if image is None:
        raise ValueError("Invalid input image")

    height, width = image.shape[:2]

    # ==========================================
    # 1. IMAGE PREPROCESSING
    # ==========================================

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # ==========================================
    # 2. FOREGROUND SEGMENTATION
    # ==========================================

    if empty_image is not None:

        if empty_image.shape != image.shape:
            raise ValueError(
                "Empty image must have the same dimensions"
            )

        # Background subtraction using LAB color
        lab = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        background = cv2.cvtColor(
            cv2.GaussianBlur(empty_image, (5, 5), 0),
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        difference = np.linalg.norm(
            lab - background,
            axis=2
        )

        mask = (
            difference > 28
        ).astype(np.uint8) * 255

    else:

        # Automatic segmentation without background image
        hsv = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2HSV
        )

        h, s, v = cv2.split(hsv)

        # Colorful objects
        colorful_mask = (
            (s >= saturation_threshold) &
            (v >= 65)
        ).astype(np.uint8) * 255

        # Bright or white objects
        bright_mask = (
            (s <= 90) &
            (v >= brightness_threshold)
        ).astype(np.uint8) * 255

        # Combine masks
        mask = cv2.bitwise_or(
            colorful_mask,
            bright_mask
        )

    # ==========================================
    # 3. MORPHOLOGICAL OPERATIONS
    # ==========================================

    # Remove small noise
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    # Close small holes
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # ==========================================
    # 4. CONTOUR DETECTION
    # ==========================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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
        aspect_ratio = max(
            w / h,
            h / w
        )

        if aspect_ratio > 5:
            continue

        # Ignore regions touching image boundaries
        if (
            x <= 2 or
            y <= 2 or
            x + w >= width - 2 or
            y + h >= height - 2
        ):
            continue

        # ======================================
        # 5. OBJECT CENTER
        # ======================================

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(
            moments["m10"] / moments["m00"]
        )

        cy = int(
            moments["m01"] / moments["m00"]
        )

        objects.append({
            "id": "",
            "pixel": [cx, cy],
            "bbox": [
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            ],
            "area_px": int(area)
        })

    # ==========================================
    # 6. SORT & ASSIGN IDs
    # ==========================================

    objects.sort(
        key=lambda obj: (
            obj["pixel"][1],
            obj["pixel"][0]
        )
    )

    for index, obj in enumerate(objects, 1):
        obj["id"] = f"obj_{index}"

    return objects, mask


def show_detections(
    image,
    objects,
    save_path=None
):
    """
    Visualize object bounding boxes and pixel centers.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.imshow(rgb)

    for obj in objects:

        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["pixel"]

        # Bounding box
        rectangle = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2
        )

        ax.add_patch(rectangle)

        # Center point
        ax.scatter(
            cx,
            cy,
            color="red",
            marker="x",
            s=40
        )

        # Object label
        ax.text(
            x1,
            max(0, y1 - 10),
            f'{obj["id"]} ({cx}, {cy})',
            color="white",
            fontsize=10,
            bbox={
                "facecolor": "black",
                "alpha": 0.8,
                "pad": 3
            }
        )

    ax.set_title(
        f"Detected Objects: {len(objects)}"
    )

    ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)


def show_mask(mask):
    """
    Visualize the binary mask used for detection.
    """

    plt.figure(figsize=(14, 10))

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title("Foreground Segmentation Mask")
    plt.axis("off")
    plt.show()


# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":

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
        min_area=1200,
        max_area_ratio=0.12,
        saturation_threshold=85,
        brightness_threshold=175
    )

    # Print JSON
    print(json.dumps(objects, indent=4))

    # Save JSON
    with open("detections.json", "w") as f:
        json.dump(objects, f, indent=4)

    # Show object detections
    show_detections(
        image,
        objects,
        save_path="detections.png"
    )

    # Show segmentation mask
    show_mask(mask)