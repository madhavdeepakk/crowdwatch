"""Drawing on video frames.

The dashboard draws zones and labels itself, as crisp vector graphics on top
of the stream. What is burned into the pixels here is what has to be: the
privacy blur, the heat map, and a small marker on each tracked person. Zone
outlines can also be burned in, for annotated video files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# BGR
LEVEL_COLOURS = [(77, 122, 31), (0, 160, 214), (10, 96, 217), (30, 38, 179)]
INK = (58, 32, 22)
WHITE = (255, 255, 255)


@dataclass
class ViewOptions:
    markers: bool = True
    heatmap: bool = False
    blur: bool = True
    zones: bool = False       # burn zone outlines into the picture


def pixelate_heads(image: np.ndarray, boxes: np.ndarray) -> None:
    """Hide faces by pixelating the top of every person box, in place."""
    import cv2

    h, w = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 12:
            continue
        hx1, hx2 = int(max(0, x1 + 0.12 * bw)), int(min(w, x2 - 0.12 * bw))
        hy1, hy2 = int(max(0, y1 - 0.03 * bh)), int(min(h, y1 + 0.24 * bh))
        if hx2 - hx1 < 4 or hy2 - hy1 < 4:
            continue
        patch = image[hy1:hy2, hx1:hx2]
        blocks = max(2, (hx2 - hx1) // 7)
        small = cv2.resize(patch, (blocks, max(2, int(blocks * patch.shape[0] / patch.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        image[hy1:hy2, hx1:hx2] = cv2.resize(small, (patch.shape[1], patch.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)


def overlay_heatmap(image: np.ndarray, grid: np.ndarray) -> None:
    import cv2

    h, w = image.shape[:2]
    heat = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 1.6)
    peak = float(heat.max())
    if peak <= 1e-4:
        return
    heat = np.clip(heat / max(peak, 0.25), 0, 1)
    big = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    colour = cv2.applyColorMap((np.clip(big, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (np.clip(big, 0, 1) ** 0.7 * 0.62)[:, :, None]
    image[:] = (image * (1 - alpha) + colour * alpha).astype(np.uint8)


def draw_markers(image: np.ndarray, boxes: np.ndarray, anchors: np.ndarray, seen: np.ndarray,
                 compact: bool) -> None:
    import cv2

    for (x1, y1, x2, y2), (ax, ay), is_seen in zip(boxes.astype(int), anchors.astype(int), seen):
        if compact:
            cv2.circle(image, (ax, ay), 11, WHITE, 2, cv2.LINE_AA)
            cv2.circle(image, (ax, ay), 11, INK, 1, cv2.LINE_AA)
            continue
        # corner ticks rather than full boxes: readable without hiding the scene
        tick = max(4, int(0.18 * min(x2 - x1, y2 - y1)))
        colour = WHITE if is_seen else (190, 190, 190)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(image, (cx, cy), (cx + dx * tick, cy), INK, 3, cv2.LINE_AA)
            cv2.line(image, (cx, cy), (cx, cy + dy * tick), INK, 3, cv2.LINE_AA)
            cv2.line(image, (cx, cy), (cx + dx * tick, cy), colour, 1, cv2.LINE_AA)
            cv2.line(image, (cx, cy), (cx, cy + dy * tick), colour, 1, cv2.LINE_AA)
        cv2.circle(image, (ax, ay), 3, INK, -1, cv2.LINE_AA)
        cv2.circle(image, (ax, ay), 2, colour, -1, cv2.LINE_AA)


def draw_zones(image: np.ndarray, engine, snapshot: dict) -> None:
    import cv2

    overlay = image.copy()
    by_id = {z["id"]: z for z in snapshot["zones"]}
    for zone in engine.zones:
        info = by_id[zone.cfg.id]
        cv2.fillPoly(overlay, [zone.polygon_px.astype(np.int32)], LEVEL_COLOURS[info["level"]])
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0, dst=image)
    for zone in engine.zones:
        info = by_id[zone.cfg.id]
        colour = LEVEL_COLOURS[info["level"]]
        pts = zone.polygon_px.astype(np.int32)
        cv2.polylines(image, [pts], True, colour, 2, cv2.LINE_AA)
        label = f"{info['name']}  {info['count']}/{info['capacity']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        y = max(y, th + 10)
        cv2.rectangle(image, (x, y - th - 10), (x + tw + 12, y), colour, -1)
        cv2.putText(image, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    INK if info["level"] == 1 else WHITE, 1, cv2.LINE_AA)
    for line in engine.lines:
        a, b = line.a.astype(int), line.b.astype(int)
        cv2.line(image, tuple(a), tuple(b), WHITE, 3, cv2.LINE_AA)
        cv2.line(image, tuple(a), tuple(b), (168, 95, 29), 1, cv2.LINE_AA)


def annotate(image: np.ndarray, engine, snapshot: dict, options: ViewOptions,
             compact_markers: bool = False) -> np.ndarray:
    """Return a copy of ``image`` with the requested overlays drawn on it."""
    out = image.copy()
    tracks = engine.tracks
    if options.blur and len(tracks) and not compact_markers:
        pixelate_heads(out, tracks.boxes)
    if options.heatmap:
        overlay_heatmap(out, engine.heatmap.normalised())
    if options.zones:
        draw_zones(out, engine, snapshot)
    if options.markers and len(tracks):
        draw_markers(out, tracks.boxes, engine.anchors, tracks.seen, compact_markers)
    return out
