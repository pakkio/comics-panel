#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('comic_panel_detector')

def detect_panels_improved(
        image_path: Path,
        out_dir: Path,
        *,
        min_panel_area_ratio: float = 0.01,
        max_panel_area_ratio: float = 0.5,
        edge_method: str = "dual",
        black_threshold: int = 60,
        white_threshold: int = 200,
        line_thickness_ratio: float = 0.005,
        morph_iterations: int = 2,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
        enable_contour_method: bool = True,
        enable_watershed_method: bool = True,
        enable_grid_fallback: bool = True,
        debug: bool = False,
        debug_viz: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, np.ndarray]]:
    debug_images = {}
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine: {image_path}")

    debug_images["original"] = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    logger.info(f"Dimensioni immagine: {w}x{h} pixel")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    debug_images["gray"] = gray.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    debug_images["enhanced"] = enhanced.copy()

    all_panels = []

    if enable_contour_method:
        logger.info("Metodo 1: Rilevamento basato sui bordi...")

        if edge_method in ["black", "dual"]:
            _, black_mask = cv2.threshold(enhanced, black_threshold, 255, cv2.THRESH_BINARY_INV)
            debug_images["black_mask"] = black_mask.copy()

            kernel_size = max(1, int(min(w, h) * line_thickness_ratio))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            black_lines = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
            debug_images["black_lines"] = black_lines.copy()

        if edge_method in ["white", "dual"]:
            _, white_mask = cv2.threshold(enhanced, white_threshold, 255, cv2.THRESH_BINARY)
            debug_images["white_mask"] = white_mask.copy()

            kernel_size = max(1, int(min(w, h) * line_thickness_ratio))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            white_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
            debug_images["white_lines"] = white_lines.copy()

        if edge_method == "canny":
            # Definisci kernel_size anche per Canny
            kernel_size = max(1, int(min(w, h) * line_thickness_ratio))
            if kernel_size % 2 == 0:
                kernel_size += 1

            edges = cv2.Canny(enhanced, 30, 100)
            debug_images["canny_edges"] = edges.copy()

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            edge_lines = cv2.dilate(edges, kernel, iterations=morph_iterations)
            debug_images["edge_lines"] = edge_lines.copy()

        if edge_method == "black":
            line_mask = black_lines
        elif edge_method == "white":
            line_mask = white_lines
        elif edge_method == "dual":
            line_mask = cv2.bitwise_or(black_lines, white_lines)
            debug_images["combined_lines"] = line_mask.copy()
        else:  # canny
            line_mask = edge_lines

        contours, hierarchy = cv2.findContours(
            line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_contours_img = img_bgr.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
        debug_images["all_contours"] = all_contours_img.copy()

        edge_panels = []
        min_area = min_panel_area_ratio * w * h
        max_area = max_panel_area_ratio * w * h

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)

                aspect_ratio = bw / bh if bh > 0 else 0
                inverse_aspect = bh / bw if bw > 0 else 0

                if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio or
                        min_aspect_ratio <= inverse_aspect <= max_aspect_ratio):

                    rect_area = bw * bh
                    area_ratio = area / rect_area if rect_area > 0 else 0

                    if area_ratio > 0.6:
                        edge_panels.append({
                            "bbox": [int(x), int(y), int(bw), int(bh)],
                            "area": int(area),
                            "method": "contour",
                            "area_ratio": float(area_ratio),
                            "aspect_ratio": float(aspect_ratio)
                        })

        all_panels.extend(edge_panels)
        logger.info(f"Rilevati {len(edge_panels)} pannelli tramite contorni")

        contour_panels_img = img_bgr.copy()
        for i, panel in enumerate(edge_panels):
            x, y, w, h = panel["bbox"]
            cv2.rectangle(contour_panels_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                contour_panels_img,
                f"C{i+1}",
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        debug_images["contour_panels"] = contour_panels_img.copy()

    if enable_watershed_method and len(all_panels) < 4:
        logger.info("Metodo 2: Applicazione watershed...")

        if 'line_mask' in locals():
            panel_mask = cv2.bitwise_not(line_mask)
        else:
            _, panel_mask = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)

        debug_images["panel_mask"] = panel_mask.copy()

        dist_transform = cv2.distanceTransform(panel_mask, cv2.DIST_L2, 5)
        debug_images["distance_transform"] = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, markers = cv2.threshold(dist_norm, 100, 255, cv2.THRESH_BINARY)

        num_labels, labels = cv2.connectedComponents(markers)

        labels = labels + 1

        if 'line_mask' in locals():
            labels[line_mask == 255] = 0

        markers = cv2.watershed(img_bgr, labels.astype(np.int32))

        markers_img = np.zeros(img_bgr.shape, dtype=np.uint8)
        for i in range(2, np.max(markers) + 1):
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            markers_img[markers == i] = color

        debug_images["watershed_markers"] = markers_img.copy()

        watershed_panels = []
        for i in range(2, np.max(markers) + 1):
            mask = np.zeros(markers.shape, dtype=np.uint8)
            mask[markers == i] = 255

            region_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if region_contours:
                cnt = max(region_contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)

                if min_area <= area <= max_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)

                    aspect_ratio = bw / bh if bh > 0 else 0
                    inverse_aspect = bh / bw if bw > 0 else 0

                    if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio or
                            min_aspect_ratio <= inverse_aspect <= max_aspect_ratio):

                        rect_area = bw * bh
                        area_ratio = area / rect_area if rect_area > 0 else 0

                        if area_ratio > 0.6:
                            watershed_panels.append({
                                "bbox": [int(x), int(y), int(bw), int(bh)],
                                "area": int(area),
                                "method": "watershed",
                                "marker_id": int(i),
                                "area_ratio": float(area_ratio),
                                "aspect_ratio": float(aspect_ratio)
                            })

        all_panels.extend(watershed_panels)
        logger.info(f"Rilevati {len(watershed_panels)} pannelli tramite watershed")

        watershed_panels_img = img_bgr.copy()
        for i, panel in enumerate(watershed_panels):
            x, y, w, h = panel["bbox"]
            cv2.rectangle(watershed_panels_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                watershed_panels_img,
                f"W{i+1}",
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        debug_images["watershed_panels"] = watershed_panels_img.copy()

    if enable_grid_fallback and len(all_panels) < 4:
        logger.info("Metodo 3: Fallback a rilevamento griglia...")

        grid_panels = detect_panels_by_projection(
            enhanced,
            min_area_ratio=min_panel_area_ratio,
            max_area_ratio=max_panel_area_ratio,
            black_threshold=black_threshold,
            white_threshold=white_threshold
        )

        all_panels.extend(grid_panels)
        logger.info(f"Rilevati {len(grid_panels)} pannelli tramite griglia")

        grid_panels_img = img_bgr.copy()
        for i, panel in enumerate(grid_panels):
            x, y, w, h = panel["bbox"]
            cv2.rectangle(grid_panels_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(
                grid_panels_img,
                f"G{i+1}",
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        debug_images["grid_panels"] = grid_panels_img.copy()

    final_panels = remove_overlapping_panels(all_panels)
    logger.info(f"Pannelli finali dopo rimozione duplicati: {len(final_panels)}")

    final_panels.sort(key=lambda p: panel_sort_key(p, row_threshold=0.4))

    final_panels = improve_panel_order(final_panels, img_bgr.shape[0], img_bgr.shape[1])

    final_panels_img = img_bgr.copy()
    for i, panel in enumerate(final_panels):
        x, y, w, h = panel["bbox"]
        method = panel.get("method", "unknown")
        color = (0, 255, 0)  # Default: verde
        if method == "watershed":
            color = (255, 0, 0)  # Blu
        elif method == "grid":
            color = (0, 165, 255)  # Arancione

        cv2.rectangle(final_panels_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            final_panels_img,
            str(i + 1),
            (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    debug_images["final_panels"] = final_panels_img.copy()

    if debug_viz:
        for name, img in debug_images.items():
            cv2.namedWindow(f"Debug: {name}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Debug: {name}", img)

        print("Premi un tasto per continuare...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if debug:
        for name, img in debug_images.items():
            debug_path = out_dir / f"{image_path.stem}_debug_{name}.jpg"
            cv2.imwrite(str(debug_path), img)
            logger.info(f"Debug image saved: {debug_path}")

    return img_bgr, final_panels, debug_images


def detect_panels_by_projection(
        gray_img: np.ndarray,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.5,
        black_threshold: int = 60,
        white_threshold: int = 200
) -> List[Dict[str, Any]]:
    h, w = gray_img.shape

    _, black_mask = cv2.threshold(gray_img, black_threshold, 255, cv2.THRESH_BINARY_INV)
    _, white_mask = cv2.threshold(gray_img, white_threshold, 255, cv2.THRESH_BINARY)

    separator_mask = cv2.bitwise_or(black_mask, white_mask)

    h_proj = np.sum(separator_mask, axis=1)
    h_proj = h_proj / w

    v_proj = np.sum(separator_mask, axis=0)
    v_proj = v_proj / h

    h_threshold = np.mean(h_proj) * 1.5
    v_threshold = np.mean(v_proj) * 1.5

    h_borders = [0]
    for i in range(1, len(h_proj) - 1):
        if h_proj[i] > h_threshold:
            start = i
            while i < len(h_proj) - 1 and h_proj[i] > h_threshold:
                i += 1
            end = i
            h_borders.append((start + end) // 2)
    h_borders.append(h - 1)

    v_borders = [0]
    for i in range(1, len(v_proj) - 1):
        if v_proj[i] > v_threshold:
            start = i
            while i < len(v_proj) - 1 and v_proj[i] > v_threshold:
                i += 1
            end = i
            v_borders.append((start + end) // 2)
    v_borders.append(w - 1)

    if len(h_borders) < 3:
        rows = 3
        h_borders = [0]
        h_borders.extend([int(h * i / rows) for i in range(1, rows)])
        h_borders.append(h - 1)

    if len(v_borders) < 3:
        cols = 2
        v_borders = [0]
        v_borders.extend([int(w * i / cols) for i in range(1, cols)])
        v_borders.append(w - 1)

    panels = []
    min_area = min_area_ratio * w * h
    max_area = max_area_ratio * w * h

    for i in range(len(h_borders) - 1):
        for j in range(len(v_borders) - 1):
            x1 = v_borders[j]
            y1 = h_borders[i]
            x2 = v_borders[j+1]
            y2 = h_borders[i+1]

            x1 = min(x1 + 1, w - 1)
            y1 = min(y1 + 1, h - 1)
            x2 = max(x2 - 1, 0)
            y2 = max(y2 - 1, 0)

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            area = width * height
            if min_area <= area <= max_area:
                panels.append({
                    "bbox": [int(x1), int(y1), int(width), int(height)],
                    "area": int(area),
                    "method": "grid",
                    "aspect_ratio": float(width / height) if height > 0 else 0
                })

    return panels


def panel_sort_key(panel: Dict[str, Any], row_threshold: float = 0.3) -> Tuple[int, int]:
    x, y, w, h = panel["bbox"]

    center_y = y + h / 2

    row = int(center_y / (h * row_threshold)) if h > 0 else 0

    return (row, x)


def improve_panel_order(panels: List[Dict[str, Any]], img_height: int, img_width: int) -> List[Dict[str, Any]]:
    if not panels:
        return []

    panel_coords = []
    for i, panel in enumerate(panels):
        x, y, w, h = panel["bbox"]
        panel_coords.append({
            "index": i,
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
            "cx": x + w // 2,
            "cy": y + h // 2,
            "panel": panel
        })

    rows = []
    current_row = [panel_coords[0]]
    vertical_overlap_threshold = 0.3

    for p in panel_coords[1:]:
        in_current_row = False
        for row_panel in current_row:
            y_overlap = min(p["y2"], row_panel["y2"]) - max(p["y1"], row_panel["y1"])
            p_height = p["y2"] - p["y1"]
            row_p_height = row_panel["y2"] - row_panel["y1"]

            if y_overlap > 0 and (y_overlap / p_height > vertical_overlap_threshold or
                                  y_overlap / row_p_height > vertical_overlap_threshold):
                in_current_row = True
                break

        if in_current_row:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]

    if current_row:
        rows.append(current_row)

    for row in rows:
        row.sort(key=lambda p: p["x1"])

    ordered_panels = []
    for row in rows:
        for p in row:
            ordered_panels.append(p["panel"])

    return ordered_panels


def remove_overlapping_panels(panels: List[Dict[str, Any]], overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    if not panels:
        return []

    sorted_panels = sorted(panels, key=lambda p: p["area"], reverse=True)

    final_panels = [sorted_panels[0]]

    for panel in sorted_panels[1:]:
        should_add = True

        for existing_panel in final_panels:
            if calculate_iou(panel["bbox"], existing_panel["bbox"]) > overlap_threshold:
                should_add = False
                break

        if should_add:
            final_panels.append(panel)

    return final_panels


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x1, box1_y1 = x1, y1
    box1_x2, box1_y2 = x1 + w1, y1 + h1

    box2_x1, box2_y1 = x2, y2
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def draw_panels(img_bgr: np.ndarray, panels: List[Dict[str, Any]]) -> np.ndarray:
    annotated = img_bgr.copy()
    for idx, panel in enumerate(panels, 1):
        x, y, w, h = panel["bbox"]

        color = (0, 255, 0)
        method = panel.get("method", "unknown")
        if method == "watershed":
            color = (255, 0, 0)
        elif method == "grid":
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            str(idx),
            (x + 4, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
def _pad_bbox(bbox, pad_ratio, img_w, img_h):
    x, y, w, h = bbox
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_w - x, w + 2 * pad_x)
    h = min(img_h - y, h + 2 * pad_y)
    return [x, y, w, h]

def _export_crops(img_bgr, panels, stem, out_dir, fmt, pad_ratio=0.05):
    H, W = img_bgr.shape[:2]
    for idx, p in enumerate(panels, 1):
        x, y, w, h = _pad_bbox(p["bbox"], pad_ratio, W, H)
        crop = img_bgr[y : y + h, x : x + w]

        filename = f"{stem}_panel_{idx:02d}.{fmt}"
        filepath = out_dir / filename

        if fmt == "jpg":
            cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rileva e estrai pannelli di fumetti con metodi avanzati"
    )
    parser.add_argument("image", type=Path, help="Percorso all'immagine di input")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Directory per le immagini annotate, ritagli e JSON",
    )
    parser.add_argument("--json", action="store_true", help="Salva metadati dei pannelli in JSON")
    parser.add_argument("--no-crops", dest="crops", action="store_false", help="Salta ritaglio pannelli")
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Formato di output per i ritagli dei pannelli (default: jpg)",
    )

    parser.add_argument(
        "--edge-method",
        choices=["white", "black", "dual", "canny"],
        default="dual",
        help="Metodo di rilevamento bordi (default: dual)",
    )
    parser.add_argument("--black-threshold", type=int, default=60,
                        help="Soglia per bordi neri (0-255)")
    parser.add_argument("--white-threshold", type=int, default=200,
                        help="Soglia per spazi bianchi (0-255)")

    parser.add_argument("--min-area-ratio", type=float, default=0.01,
                        help="Rapporto minimo area pannello/pagina")
    parser.add_argument("--max-area-ratio", type=float, default=0.5,
                        help="Rapporto massimo area pannello/pagina")

    parser.add_argument("--no-contour", dest="enable_contour", action="store_false",
                        help="Disattiva rilevamento basato su contorni")
    parser.add_argument("--no-watershed", dest="enable_watershed", action="store_false",
                        help="Disattiva rilevamento basato su watershed")
    parser.add_argument("--no-grid", dest="enable_grid", action="store_false",
                        help="Disattiva fallback a rilevamento griglia")

    parser.add_argument("--debug", action="store_true", help="Salva immagini di debug")
    parser.add_argument("--debug-viz", action="store_true", help="Mostra immagini di debug a schermo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output dettagliato")
    parser.add_argument("--line-thickness-ratio", type=float, default=0.005,
                        help="Spessore della dilatazione (rapporto rispetto al lato più corto)")
    parser.add_argument("--morph-iterations", type=int, default=2,
                        help="Numero di iterazioni di dilatazione/chiusura")
    parser.add_argument("--pad", type=float, default=0.05,
                        help="Extra padding ratio added to each side of a panel (default 0.05 = 5%)")


    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    args.output.mkdir(parents=True, exist_ok=True)

    try:
        img_bgr, panels, _ = detect_panels_improved(
            args.image,
            args.output,
            min_panel_area_ratio=args.min_area_ratio,
            max_panel_area_ratio=args.max_area_ratio,
            line_thickness_ratio=args.line_thickness_ratio,
            morph_iterations=args.morph_iterations,
            edge_method=args.edge_method,
            black_threshold=args.black_threshold,
            white_threshold=args.white_threshold,
            enable_contour_method=args.enable_contour,
            enable_watershed_method=args.enable_watershed,
            enable_grid_fallback=args.enable_grid,
            debug=args.debug,
            debug_viz=args.debug_viz,
        )

        annotated = draw_panels(img_bgr, panels)
        ann_path = args.output / f"{args.image.stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info(f"Immagine annotata salvata in {ann_path}")

        if args.crops and panels:
            _export_crops(img_bgr, panels, args.image.stem, args.output, args.format, pad_ratio=args.pad)

            logger.info(f"Salvati {len(panels)} ritagli di pannelli in formato {args.format.upper()} in {args.output}")
        elif args.crops and not panels:
            logger.warning("Nessun pannello trovato da ritagliare!")

        if args.json:
            json_path = args.output / f"{args.image.stem}_panels.json"
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(panels, fp, indent=2)
            logger.info(f"Metadati JSON scritti in {json_path}")

    except Exception as e:
        logger.error(f"Errore durante l'elaborazione: {str(e)}")
        raise


if __name__ == "__main__":
    main()
