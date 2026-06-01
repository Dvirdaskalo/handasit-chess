import cv2
import json
import numpy as np
import argparse
import sys

GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


def load_board_transform():
    with open("board_transform.json") as f:
        d = json.load(f)
    M = np.array(d["M"], np.float32)
    size = tuple(d["size"])
    cell = size[0] // 8
    return M, size, cell


def load_sticker_colors():
    with open("sticker_colors.json") as f:
        return json.load(f)["pieces"]


def hue_distance(h1, h2):
    """Circular distance on [0, 180) hue wheel."""
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


SAT_MASK_THRESHOLD = 85    # include lower-saturation stickers (bishops S≈85-92)
MIN_PIXELS = 500           # minimum qualifying pixels across pooled frames

def identify_square(crop, pieces):
    """Return piece label for a square crop, or 'empty'."""
    if crop.size == 0:
        return "empty"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].flatten()
    s = hsv[:, :, 1].flatten()

    sat_mask = s > SAT_MASK_THRESHOLD
    n = int(np.sum(sat_mask))
    if n < 20:
        return "empty"

    median_h = int(np.median(h[sat_mask]))
    median_s = int(np.median(s[sat_mask]))

    best_label = "empty"
    best_dist = float("inf")
    for label, info in pieces.items():
        if median_s < info.get("s_min", 60):
            continue
        dist = hue_distance(median_h, info["h"])
        tol = info.get("h_tol", 18)
        if dist < tol and dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label


def print_board(board_state):
    lines = [CLEAR]
    lines.append("   a    b    c    d    e    f    g    h")
    lines.append("  " + "─" * 37)

    for row in range(8, 0, -1):
        line = f"{row} │"
        for col in range(8):
            square = f"{chr(ord('a') + col)}{row}"
            piece = board_state.get(square, "empty")
            if piece == "empty":
                line += "  .  "
            elif piece.startswith("w"):
                line += f" {GREEN}{piece:2s}{RESET}  "
            else:
                line += f" {BLUE}{piece:2s}{RESET}  "
        line += "│"
        lines.append(line)

    lines.append("  " + "─" * 37)

    counts = {}
    for piece in board_state.values():
        if piece != "empty":
            counts[piece] = counts.get(piece, 0) + 1

    if counts:
        parts = []
        for piece, cnt in sorted(counts.items()):
            color = GREEN if piece.startswith("w") else BLUE
            parts.append(f"{color}{piece}{RESET}×{cnt}")
        lines.append("\nPieces: " + "  ".join(parts))
    else:
        lines.append("\n(no pieces detected)")

    lines.append("\nPress 'q' to quit  |  'r' to reload sticker_colors.json")
    print("\n".join(lines), end="", flush=True)


def draw_overlay(warped, board_state, cell):
    display = warped.copy()
    for r in range(8):
        for c in range(8):
            square = f"{chr(ord('a') + c)}{8 - r}"
            piece = board_state.get(square, "empty")
            cv2.rectangle(display,
                          (c * cell, r * cell),
                          ((c + 1) * cell, (r + 1) * cell),
                          (180, 180, 180), 1)
            if piece != "empty":
                color = (0, 200, 0) if piece.startswith("w") else (200, 100, 0)
                cv2.putText(display, piece,
                            (c * cell + 8, r * cell + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return display


def main():
    parser = argparse.ArgumentParser(description="Chess sticker piece detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-display", action="store_true",
                        help="Skip OpenCV window (terminal only)")
    args = parser.parse_args()

    try:
        M, size, cell = load_board_transform()
    except FileNotFoundError:
        print("board_transform.json not found. Run calibrate_board.py first.")
        sys.exit(1)

    try:
        pieces = load_sticker_colors()
    except FileNotFoundError:
        print("sticker_colors.json not found. Run calibrate_stickers.py first.")
        sys.exit(1)

    print(f"Loaded {len(pieces)} piece colors: {list(pieces.keys())}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)

    if not args.no_display:
        cv2.namedWindow("Chess Detection")

    POOL_FRAMES = 6      # frames to pool per detection cycle
    DETECT_EVERY = 3     # run detection every N display frames
    warped_pool = []
    frame_count = 0
    board_state = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        try:
            warped = cv2.warpPerspective(frame, M, size)
        except Exception:
            continue

        warped_pool.append(warped)
        if len(warped_pool) > POOL_FRAMES:
            warped_pool.pop(0)

        frame_count += 1
        if frame_count % DETECT_EVERY == 0 and len(warped_pool) >= 2:
            pad = int(cell * 0.28)
            for r in range(8):
                for c in range(8):
                    # Pool pixels from all buffered frames
                    all_h, all_s = [], []
                    for wf in warped_pool:
                        crop = wf[r * cell + pad:(r + 1) * cell - pad,
                                  c * cell + pad:(c + 1) * cell - pad]
                        if crop.size == 0:
                            continue
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        h = hsv[:, :, 0].flatten()
                        s = hsv[:, :, 1].flatten()
                        mask = s > SAT_MASK_THRESHOLD
                        if np.sum(mask) >= 3:
                            all_h.extend(h[mask].tolist())
                            all_s.extend(s[mask].tolist())
                    square = f"{chr(ord('a') + c)}{8 - r}"
                    if len(all_h) < MIN_PIXELS:
                        board_state[square] = "empty"
                        continue
                    median_h = int(np.median(all_h))
                    median_s = int(np.median(all_s))
                    best_label = "empty"
                    best_dist = float("inf")
                    for label, info in pieces.items():
                        if median_s < info.get("s_min", 60):
                            continue
                        dist = hue_distance(median_h, info["h"])
                        tol = info.get("h_tol", 18)
                        if dist < tol and dist < best_dist:
                            best_dist = dist
                            best_label = label
                    board_state[square] = best_label

        print_board(board_state)

        if not args.no_display:
            cv2.imshow("Chess Detection", draw_overlay(warped, board_state, cell))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            try:
                pieces = load_sticker_colors()
                print(f"\nReloaded: {list(pieces.keys())}")
            except Exception as e:
                print(f"\nReload failed: {e}")

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("\n" + RESET)


if __name__ == "__main__":
    main()
