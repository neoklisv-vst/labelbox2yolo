#!/usr/bin/env python3
import os
from pathlib import Path
import ndjson
import requests
import cv2
import numpy as np
from PIL import Image

# --- helper: make dirs like your utils.make_dirs ---
def make_dirs(stem: str) -> Path:
    d = Path(stem)
    d.mkdir(parents=True, exist_ok=True)
    return d

def yolo_write_segments(label_path: Path, segments):
    """
    segments: list of tuples (class_id, [(x1,y1), (x2,y2), ...]) with coords normalized [0,1]
    Writes one row per instance: cls x1 y1 x2 y2 ... xn yn
    """
    with open(label_path, "w") as f:
        for cls_id, pts in segments:
            # Ultralytics expects at least 3 points (polygon)
            if len(pts) < 3:
                continue
            flat = " ".join([f"{x:.6f} {y:.6f}" for (x, y) in pts])
            f.write(f"{int(cls_id)} {flat}\n")

def contours_from_mask(binary_mask, approx_epsilon_frac=0.002):
    """
    Extract external contours from a binary mask (uint8 0/255).
    Returns list of Nx2 float arrays of (x,y) in pixel coords.
    """
    # Ensure binary as 0/255
    bm = (binary_mask > 0).astype(np.uint8) * 255
    # Find contours
    contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 10:  # tiny speck filter
            continue
        # Optional polygon simplification
        eps = approx_epsilon_frac * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)
        c = c.reshape(-1, 2).astype(float)  # (N,2) in (x,y) pixel coordinates
        if len(c) >= 3:
            polys.append(c)
    return polys

def normalize_polygon(poly_xy, width, height):
    """
    Convert pixel coords to normalized [0,1] YOLO order (x/width, y/height).
    """
    norm = []
    for (x, y) in poly_xy:
        nx = np.clip(x / width, 0.0, 1.0)
        ny = np.clip(y / height, 0.0, 1.0)
        norm.append((float(nx), float(ny)))
    return norm

def get_masks_as_yolo_segments(
    PROJECT_ID: str,
    api_key: str,
    class_indices: dict,
    export_filename: str = "k4a-seg.ndjson",
    out_root: str | Path = None,
    min_points: int = 6,
):
    """
    Reads Labelbox NDJSON export and writes Ultralytics YOLO segmentation labels.

    class_indices: mapping from Labelbox object 'name' -> integer class id (0..N-1 recommended)
    """
    with open(export_filename) as f:
        data = ndjson.load(f)

    file = Path(export_filename)
    save_dir = Path(out_root) if out_root else make_dirs(file.stem)

    images_dir = save_dir / "images"
    labels_dir = save_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": api_key}

    for i, d in enumerate(data):
        # --- fetch image ---
        im_path = d["data_row"]["row_data"]
        image_name = d["data_row"]["external_id"]
        pil_im = Image.open(requests.get(im_path, stream=True).raw if str(im_path).startswith("http") else im_path)
        width, height = pil_im.size

        image_path = images_dir / image_name
        pil_im.save(image_path, quality=95, subsampling=0)

        # --- build semantic mask per class ---
        # Labelbox media size is in data[i]['media_attributes']
        H = d["media_attributes"]["height"]
        W = d["media_attributes"]["width"]

        # collect instance polygons -> list of (cls_id, [(x,y),...]) normalized
        segments_out = []

        objects = d["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"]
        # For each Labelbox object (instance), download its mask and convert to polygon
        for obj in objects:
            name = obj["name"]
            if name not in class_indices:
                continue
            cls_id = class_indices[name]
            url = obj["mask"]["url"]

            # Download instance mask (binary)
            with requests.get(url, headers=headers, stream=True) as r:
                r.raw.decode_content = True
                mask_bytes = np.asarray(bytearray(r.raw.read()), dtype="uint8")
                mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)

            # Sanity: ensure mask size matches media attrs
            if mask_img is None:
                print(f"[WARN] Could not decode mask for {image_name} / class {name}")
                continue
            if mask_img.shape[:2] != (H, W):
                mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

            # Extract instance contours (each object row is already an instance)
            polys_px = contours_from_mask(mask_img)

            # If Labelbox mask is a single blob, you'll usually get one polygon.
            for poly in polys_px:
                if len(poly) < max(3, min_points // 2):
                    # Optional: densify a bit by interpolating to ensure enough points
                    pass
                # Normalize to [0,1] in YOLO (x/width, y/height)
                norm_poly = normalize_polygon(poly, W, H)
                # YOLO requires an even number of values (x,y pairs) and >= 3 points
                if len(norm_poly) >= 3:
                    segments_out.append((cls_id, norm_poly))

        # --- write YOLO label file ---
        label_path = labels_dir / (Path(image_name).with_suffix(".txt").name)
        yolo_write_segments(label_path, segments_out)

        print(f"Wrote {label_path} with {len(segments_out)} instance(s)")

if __name__ == "__main__":
    # --- user config ---
    # Example:
    #project_id = "ckxxxxxxxxxxxxxxxxxxx"
     api_key = "xxxxxxxxxxxxxxxxxxxxxxxxx"   # include the "Bearer " prefix
    
    # Map Labelbox object names -> YOLO class IDs (start at 0 if training from scratch)
    # e.g., {"bed": 0, "person_in_bed": 1}
    class_indices = {"bed": 0}

    # path to your Labelbox export (NDJSON)
    export_filename = "k4a-seg.ndjson"

    get_masks_as_yolo_segments(
        PROJECT_ID=project_id,
        api_key=api_key,
        class_indices=class_indices,
        export_filename=export_filename,
    )
