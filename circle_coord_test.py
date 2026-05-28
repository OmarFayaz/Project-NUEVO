import cv2
import numpy as np
import time
import math

FRAME_W = 640
FRAME_H = 480
FRAME_CX = FRAME_W / 2
FRAME_CY = FRAME_H / 2

OUTPUT_PATH = "/runtime_output/vision/circle_centers_latest.jpg"

# Circle thresholds
MIN_AREA = 150
MAX_AREA = 12000
MIN_CIRCULARITY = 0.50
MIN_SIZE = 10
MAX_SIZE = 170

# Cardboard/background validation
PAD = 22
MIN_CARDBOARD_RATIO = 0.18

# Stability filter
MATCH_RADIUS_PX = 45
CONFIRM_FRAMES = 4
MAX_MISSED_FRAMES = 3

# Panel style
PANEL_BG = (180, 245, 245)   # light yellow-ish in BGR
PANEL_ALPHA = 0.88
PANEL_TEXT = (0, 0, 0)       # black
PANEL_MARGIN = 10
LINE_HEIGHT = 22

tracks = []
next_track_id = 0


def make_cardboard_mask(hsv):
    lower = np.array([8, 35, 55])
    upper = np.array([32, 190, 230])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def has_cardboard_background(cardboard_mask, color_mask, x, y, w, h):
    x1 = max(0, x - PAD)
    y1 = max(0, y - PAD)
    x2 = min(FRAME_W, x + w + PAD)
    y2 = min(FRAME_H, y + h + PAD)

    cardboard_roi = cardboard_mask[y1:y2, x1:x2]
    color_roi = color_mask[y1:y2, x1:x2]

    surrounding = cv2.bitwise_and(cardboard_roi, cv2.bitwise_not(color_roi))

    total_pixels = surrounding.size
    if total_pixels == 0:
        return False, 0.0

    ratio = cv2.countNonZero(surrounding) / float(total_pixels)
    return ratio >= MIN_CARDBOARD_RATIO, ratio


def detect_color_blobs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cardboard_mask = make_cardboard_mask(hsv)

    red_mask = (
        cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255])) |
        cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
    )

    green_mask = cv2.inRange(
        hsv,
        np.array([40, 70, 70]),
        np.array([90, 255, 255])
    )

    kernel = np.ones((5, 5), np.uint8)
    results = []

    for color_name, raw_mask in [("red", red_mask), ("green", green_mask)]:
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
                continue

            aspect = w / h
            if aspect < 0.55 or aspect > 1.8:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue

            ok_bg, bg_ratio = has_cardboard_background(cardboard_mask, mask, x, y, w, h)
            if not ok_bg:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            results.append({
                "color": color_name,
                "cx": cx,
                "cy": cy,
                "err_x": cx - FRAME_CX,
                "err_y": cy - FRAME_CY,
                "area": area,
                "circ": circularity,
                "box": (x, y, w, h),
                "diameter_px": 0.5 * (w + h),
                "width_px": w,
                "height_px": h,
                "cardboard_ratio": bg_ratio,
            })

    results.sort(key=lambda d: (d["cy"], d["cx"]))
    return results


def dist_px(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def update_tracks(detections):
    global next_track_id, tracks

    for t in tracks:
        t["matched"] = False

    for d in detections:
        candidates = [
            t for t in tracks
            if t["color"] == d["color"]
            and not t["matched"]
            and dist_px(t, d) < MATCH_RADIUS_PX
        ]

        if candidates:
            t = min(candidates, key=lambda tr: dist_px(tr, d))
            alpha = 0.45

            t["cx"] = alpha * d["cx"] + (1 - alpha) * t["cx"]
            t["cy"] = alpha * d["cy"] + (1 - alpha) * t["cy"]

            for key in [
                "err_x", "err_y", "area", "circ", "box",
                "diameter_px", "width_px", "height_px", "cardboard_ratio"
            ]:
                t[key] = d[key]

            t["err_x"] = t["cx"] - FRAME_CX
            t["err_y"] = t["cy"] - FRAME_CY
            t["hits"] += 1
            t["misses"] = 0
            t["matched"] = True
        else:
            new_track = d.copy()
            new_track["id"] = next_track_id
            new_track["hits"] = 1
            new_track["misses"] = 0
            new_track["matched"] = True
            tracks.append(new_track)
            next_track_id += 1

    for t in tracks:
        if not t["matched"]:
            t["misses"] += 1

    tracks = [t for t in tracks if t["misses"] <= MAX_MISSED_FRAMES]

    stable = [
        t for t in tracks
        if t["hits"] >= CONFIRM_FRAMES and t["misses"] == 0
    ]

    stable.sort(key=lambda d: (d["cy"], d["cx"]))
    return stable


def draw_info_panel(out, stable_tracks):
    lines = ["Stable cardboard circles"]

    if not stable_tracks:
        lines.append("None")
    else:
        for d in stable_tracks:
            line = (
                f"ID {d['id']}  {d['color']}  "
                f"({int(round(d['cx']))},{int(round(d['cy']))})  "
                f"D={int(round(d['diameter_px']))} px"
            )
            lines.append(line)

    panel_w = 340
    panel_h = PANEL_MARGIN * 2 + LINE_HEIGHT * len(lines)

    x1 = 8
    y2 = FRAME_H - 8
    x2 = min(FRAME_W - 8, x1 + panel_w)
    y1 = max(8, y2 - panel_h)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, out, 1 - PANEL_ALPHA, 0, out)

    for i, line in enumerate(lines):
        tx = x1 + PANEL_MARGIN
        ty = y1 + PANEL_MARGIN + (i + 1) * LINE_HEIGHT - 6
        cv2.putText(
            out,
            line,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            PANEL_TEXT,
            1,
            cv2.LINE_AA
        )

    return out


def draw_results(frame, stable_tracks):
    out = frame.copy()

    cv2.drawMarker(
        out,
        (int(FRAME_CX), int(FRAME_CY)),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=2,
    )

    for d in stable_tracks:
        cx = int(d["cx"])
        cy = int(d["cy"])
        x, y, w, h = d["box"]

        color = (0, 0, 255) if d["color"] == "red" else (0, 255, 0)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        cv2.circle(out, (cx, cy), 7, color, -1)

        # Only draw ID near the detected circle
        label = f"ID {d['id']}"
        cv2.putText(
            out,
            label,
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    out = draw_info_panel(out, stable_tracks)
    return out


def main():
    cap = cv2.VideoCapture("/dev/video10")

    if not cap.isOpened():
        print("ERROR: Could not open /dev/video10")
        print("Stop vision_debug.launch.py first.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print("Reading /dev/video10. Press Ctrl+C to stop.")
    print("Open image:", OUTPUT_PATH)
    print("Showing IDs on circles, and full coordinates/diameters in bottom-left panel.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                time.sleep(0.2)
                continue

            raw_detections = detect_color_blobs(frame)
            stable_tracks = update_tracks(raw_detections)

            out = draw_results(frame, stable_tracks)
            cv2.imwrite(OUTPUT_PATH, out)

            print("-" * 80)
            print(f"raw_on_cardboard={len(raw_detections)} stable={len(stable_tracks)} active_tracks={len(tracks)}")

            if not stable_tracks:
                print("No stable cardboard-backed red/green circles found")

            for d in stable_tracks:
                print(
                    f"ID={d['id']:02d} {d['color']:5s} "
                    f"center=({d['cx']:.1f}, {d['cy']:.1f}) "
                    f"diam={d['diameter_px']:.1f}px "
                    f"size=({d['width_px']}x{d['height_px']}) "
                    f"hits={d['hits']} "
                    f"cardboard={d['cardboard_ratio']:.2f} "
                    f"area={d['area']:.0f} circ={d['circ']:.2f}"
                )

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nStopped.")

    cap.release()


if __name__ == "__main__":
    main()
