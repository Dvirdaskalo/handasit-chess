import os
import cv2
import json
import numpy as np

# Suppress Qt font/logging noise that can destabilize the window
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.*.debug=false")

SAMPLE_FRAMES = 5    # frames to pool per sample for stable hue
PANEL_H = 240        # height of checklist panel below the board
DETECT_EVERY = 3     # run live detection overlay only every N frames

# Key → piece label  (lowercase = white, uppercase = black)
KEY_MAP = {
    ord('k'): 'wk', ord('q'): 'wq', ord('r'): 'wr',
    ord('b'): 'wb', ord('n'): 'wn', ord('p'): 'wp',
    ord('K'): 'bk', ord('Q'): 'bq', ord('R'): 'br',
    ord('B'): 'bb', ord('N'): 'bn', ord('P'): 'bp',
}

WHITE_PIECES = ['wk', 'wq', 'wr', 'wb', 'wn', 'wp']
BLACK_PIECES = ['bk', 'bq', 'br', 'bb', 'bn', 'bp']
PIECE_NAMES = {
    'wk': 'King', 'wq': 'Queen', 'wr': 'Rook',
    'wb': 'Bishop', 'wn': 'Knight', 'wp': 'Pawn',
    'bk': 'King', 'bq': 'Queen', 'br': 'Rook',
    'bb': 'Bishop', 'bn': 'Knight', 'bp': 'Pawn',
}

try:
    open('board_transform.json').close()
except FileNotFoundError:
    print('board_transform.json not found. Run calibrate_board.py first.')
    raise SystemExit(1)

with open('board_transform.json') as f:
    d = json.load(f)
M = np.array(d['M'], np.float32)
size = tuple(d['size'])
CELL = size[0] // 8

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera.')
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency

calibration = {}
last_added = None
unsaved = False
frame_buffer = []
frame_count = 0

# Pending state: None = idle, else (row, col, sampled_h, sampled_s, sampled_v)
pending = None


# ── helpers ──────────────────────────────────────────────────────────────────

def hue_distance(h1, h2):
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


def hsv_to_bgr(h, s, v):
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    return tuple(int(x) for x in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def sample_square(frames, row, col):
    """Pool HSV pixels from buffered frames and return (h, s, v) medians or None."""
    pad = int(CELL * 0.35)
    all_h, all_s, all_v = [], [], []
    for buf in frames:
        crop = buf[row * CELL + pad:(row + 1) * CELL - pad,
                   col * CELL + pad:(col + 1) * CELL - pad]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].flatten()
        s = hsv[:, :, 1].flatten()
        v = hsv[:, :, 2].flatten()
        mask = s > 60
        if np.sum(mask) >= 5:
            all_h.extend(h[mask].tolist())
            all_s.extend(s[mask].tolist())
            all_v.extend(v[mask].tolist())
    if len(all_h) < 10:
        return None
    return int(np.median(all_h)), int(np.median(all_s)), int(np.median(all_v))


def detect_label(crop):
    if crop.size == 0:
        return 'empty'
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].flatten()
    h = hsv[:, :, 0].flatten()
    mask = s > 60
    if np.sum(mask) < 20:
        return 'empty'
    median_h = int(np.median(h[mask]))
    best, best_dist = 'empty', float('inf')
    for label, info in calibration.items():
        dist = hue_distance(median_h, info['h'])
        if dist < info.get('h_tol', 18) and dist < best_dist:
            best_dist = dist
            best = label
    return best


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_checklist(panel):
    panel[:] = (28, 28, 28)
    done = len(calibration)
    total = 12
    col_x = [10, size[0] // 2 + 10]

    # Header
    status = f'Progress: {done}/{total}'
    if unsaved:
        status += '  *UNSAVED*'
    color = (0, 220, 220) if unsaved else (140, 220, 140)
    cv2.putText(panel, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(panel, 's=save  u=undo  q=quit',
                (size[0] - 210, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)

    for col_idx, pieces in enumerate([WHITE_PIECES, BLACK_PIECES]):
        x0 = col_x[col_idx]
        header = 'WHITE (lowercase key)' if col_idx == 0 else 'BLACK (Shift+key)'
        cv2.putText(panel, header, (x0, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        for row_idx, label in enumerate(pieces):
            y = 64 + row_idx * 28
            key_char = label[1].upper() if col_idx == 1 else label[1]
            if label in calibration:
                info = calibration[label]
                bgr = hsv_to_bgr(info['h'], info['s'], info['v'])
                cv2.rectangle(panel, (x0, y - 14), (x0 + 20, y + 8), bgr, -1)
                cv2.rectangle(panel, (x0, y - 14), (x0 + 20, y + 8), (200, 200, 200), 1)
                text_color = (230, 230, 230)
                tick = ' v'
            else:
                cv2.rectangle(panel, (x0, y - 14), (x0 + 20, y + 8), (55, 55, 55), -1)
                cv2.rectangle(panel, (x0, y - 14), (x0 + 20, y + 8), (90, 90, 90), 1)
                text_color = (95, 95, 95)
                tick = ''
            name = PIECE_NAMES[label]
            cv2.putText(panel, f'[{key_char}] {label} {name}{tick}',
                        (x0 + 26, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)


def draw_pending_overlay(panel, h, s, v):
    """Replace checklist with key-selection prompt while waiting for piece assignment."""
    panel[:] = (20, 20, 45)  # dark blue tint to signal "waiting for input"

    # Sampled color swatch
    bgr = hsv_to_bgr(h, s, v)
    cv2.rectangle(panel, (8, 8), (70, 70), bgr, -1)
    cv2.rectangle(panel, (8, 8), (70, 70), (200, 200, 200), 2)
    cv2.putText(panel, f'H={h} S={s} V={v}', (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(panel, 'Esc = cancel', (80, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    cv2.putText(panel, 'Press key to assign piece:',
                (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 220), 1)

    # Two rows of shortcuts
    row1 = [('k', 'wk King'), ('q', 'wq Queen'), ('r', 'wr Rook'),
            ('b', 'wb Bishop'), ('n', 'wn Knight'), ('p', 'wp Pawn')]
    row2 = [('K', 'bk King'), ('Q', 'bq Queen'), ('R', 'br Rook'),
            ('B', 'bb Bishop'), ('N', 'bn Knight'), ('P', 'bp Pawn')]

    for row_idx, row in enumerate([row1, row2]):
        y = 118 + row_idx * 30
        prefix = 'White:' if row_idx == 0 else 'Black:'
        cv2.putText(panel, prefix, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        for i, (key, label) in enumerate(row):
            x = 65 + i * 122
            already = label[:2] in calibration
            color = (100, 200, 100) if not already else (80, 130, 80)
            cv2.putText(panel, f'[{key}]{label}', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def draw_board(warped, detections, highlight_sq):
    board = warped.copy()

    # Live detection tint
    if detections:
        overlay = board.copy()
        for r, c, label in detections:
            info = calibration.get(label)
            if info:
                bgr = hsv_to_bgr(info['h'], info['s'], info['v'])
                cv2.rectangle(overlay, (c * CELL, r * CELL),
                              ((c + 1) * CELL, (r + 1) * CELL), bgr, -1)
        cv2.addWeighted(overlay, 0.28, board, 0.72, 0, board)
        for r, c, label in detections:
            cv2.putText(board, label, (c * CELL + 8, r * CELL + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Grid + square names
    for i in range(9):
        cv2.line(board, (i * CELL, 0), (i * CELL, 8 * CELL), (150, 150, 150), 1)
        cv2.line(board, (0, i * CELL), (8 * CELL, i * CELL), (150, 150, 150), 1)
    for r in range(8):
        for c in range(8):
            cv2.putText(board, f'{chr(ord("a")+c)}{8-r}',
                        (c * CELL + 3, r * CELL + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130, 130, 130), 1)

    # Highlight pending square
    if highlight_sq:
        r, c = highlight_sq
        cv2.rectangle(board, (c * CELL, r * CELL),
                      ((c + 1) * CELL, (r + 1) * CELL), (0, 255, 255), 3)

    cv2.putText(board, 'Click a square with a sticker to sample it',
                (5, size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    return board


# ── mouse callback ────────────────────────────────────────────────────────────

clicked_sq = None

def mouse_callback(event, x, y, flags, param):
    global clicked_sq
    if event == cv2.EVENT_LBUTTONDOWN and y < size[1] and pending is None:
        col = x // CELL
        row = y // CELL
        if 0 <= row < 8 and 0 <= col < 8:
            clicked_sq = (row, col)


cv2.namedWindow('Calibrate Stickers')
cv2.setMouseCallback('Calibrate Stickers', mouse_callback)

print('\n=== Sticker Color Calibration ===')
print('1. Place ONE piece on any square, sticker facing up')
print('2. Click its square in the window')
print('3. Press a key to assign the piece type:')
print('   lowercase (k/q/r/b/n/p) = white piece')
print('   uppercase (K/Q/R/B/N/P) = black piece')
print('   Esc = cancel')
print('\nKeys:  s = save   u = undo last   q = quit\n')

# ── main loop ─────────────────────────────────────────────────────────────────

detections = []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    try:
        warped = cv2.warpPerspective(frame, M, size)
    except Exception:
        continue

    frame_buffer.append(warped.copy())
    if len(frame_buffer) > SAMPLE_FRAMES:
        frame_buffer.pop(0)

    frame_count += 1

    # Process a click: sample the square
    if clicked_sq is not None and pending is None:
        row, col = clicked_sq
        clicked_sq = None
        result = sample_square(frame_buffer, row, col)
        if result is None:
            print(f'Too few saturated pixels in {chr(ord("a")+col)}{8-row}. '
                  f'Is there a bright sticker there?')
        else:
            h, s, v = result
            bgr = hsv_to_bgr(h, s, v)
            print(f'Sampled {chr(ord("a")+col)}{8-row}: '
                  f'H={h} S={s} V={v}  approx RGB({bgr[2]},{bgr[1]},{bgr[0]})')
            pending = (row, col, h, s, v)

    # Live detection (throttled)
    if calibration and frame_count % DETECT_EVERY == 0 and pending is None:
        detections = []
        pad = int(CELL * 0.35)
        for r in range(8):
            for c in range(8):
                crop = warped[r * CELL + pad:(r + 1) * CELL - pad,
                              c * CELL + pad:(c + 1) * CELL - pad]
                label = detect_label(crop)
                if label != 'empty':
                    detections.append((r, c, label))

    # Build display
    highlight = (pending[0], pending[1]) if pending else None
    board_img = draw_board(warped, detections if pending is None else [], highlight)
    panel = np.zeros((PANEL_H, size[0], 3), dtype=np.uint8)

    if pending:
        draw_pending_overlay(panel, pending[2], pending[3], pending[4])
    else:
        draw_checklist(panel)

    canvas = np.vstack([board_img, panel])
    cv2.imshow('Calibrate Stickers', canvas)

    key = cv2.waitKey(1) & 0xFF

    # --- key handling ---
    if pending is not None:
        if key == 27:  # Esc = cancel
            pending = None
            print('Cancelled.')
        elif key in KEY_MAP:
            label = KEY_MAP[key]
            row, col, h, s, v = pending
            calibration[label] = {'h': h, 's': s, 'v': v, 'h_tol': 18, 's_min': 60}
            last_added = label
            unsaved = True
            pending = None
            print(f'  Assigned: {label} ({PIECE_NAMES[label]})  '
                  f'H={h}  ({len(calibration)}/12 done)\n')
    else:
        if key == ord('q'):
            if unsaved and calibration:
                print('Unsaved calibration. Press s then q to save+quit, or q again to discard.')
                # second q check handled next iteration — just break for now
                # to avoid input() blocking: user must press s manually
            break
        elif key == ord('s'):
            if calibration:
                with open('sticker_colors.json', 'w') as f:
                    json.dump({'pieces': calibration}, f, indent=2)
                unsaved = False
                print(f'Saved {len(calibration)} piece colors to sticker_colors.json')
            else:
                print('Nothing to save yet.')
        elif key == ord('u'):
            if last_added and last_added in calibration:
                del calibration[last_added]
                print(f'Undone: removed {last_added!r}')
                last_added = None
                unsaved = bool(calibration)
            else:
                print('Nothing to undo.')

cap.release()
cv2.destroyAllWindows()
print('Done.')
