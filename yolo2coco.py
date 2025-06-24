import os
import json
import cv2
from glob import glob

def convert_yolo_to_coco(images_dir, output_json_path, class_list):
    images = []
    annotations = []
    categories = []
    ann_id = 1
    img_id = 1

    for i, class_name in enumerate(class_list):
        categories.append({
            "id": i,
            "name": class_name,
            "supercategory": "object"
        })

    label_files = glob(os.path.join(images_dir, "*.txt"))

    print(f"Found {len(label_files)} label files")

    for label_file in label_files:
        base_filename = os.path.splitext(os.path.basename(label_file))[0]

        # Try both .jpg and .png extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(images_dir, base_filename + ext)
            if os.path.exists(image_path):
                break
        else:
            print(f"⚠️ Skipping: No image found for label {label_file}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue

        height, width = img.shape[:2]

        images.append({
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": img_id
        })

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, w, h = map(float, parts)
            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w = w * width
            h = h * height

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(class_id),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    print(f"✅ Processed {img_id - 1} images and {ann_id - 1} annotations")

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)

# Example usage:
convert_yolo_to_coco(
    images_dir = r"C:\Users\karishma.thumu\DETR\images\val",
    output_json_path = r"C:\Users\karishma.thumu\DETR\images\val\annotations.json",
    class_list = ["Wood", "HDPEPP-Flake-White-NonTransparent-Cap"]
)
