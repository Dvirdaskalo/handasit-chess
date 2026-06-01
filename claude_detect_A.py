#!/usr/bin/env python3
"""
Chess piece detection via Claude Opus Vision — Architecture A (speed-optimized).

Differences from claude_detect.py:
  - Prompt restructured so the cached prefix is:
        <static instructions> + <baseline label> + <baseline image>
    A `cache_control` breakpoint sits on the baseline image (1h TTL).
    Volatile content (the current image + tail prompt) lands after the
    breakpoint and is processed fresh each call.
  - `effort` lowered from "high" → "medium".
  - JPEG quality 95 → 88 (smaller upload, no visible quality loss).
  - Camera capture trimmed: 5 frames + 6 flush (was 8 + 10).
  - New flag `--fresh-baseline`: capture a new baseline AND run detection in
    one command. Use this when the lighting changes mid-game; otherwise the
    cached baseline is reused for an hour.

Workflow:
  1. Put board in starting position, run once per game:
        python claude_detect_A.py --register
  2. After moving pieces:
        python claude_detect_A.py
  3. If lighting changes or you want to refresh:
        python claude_detect_A.py --fresh-baseline
"""

import cv2
import json
import re
import sys
import base64
import argparse
import numpy as np
import anthropic

GREEN = "\033[32m"
BLUE  = "\033[34m"
RESET = "\033[0m"

BASELINE_FILE = "baseline.jpg"

VALID_CODES = {"wr","wn","wb","wq","wk","wp","br","bn","bb","bq","bk","bp"}

# Coordinate system used by board_transform.json:
#   col 0 = h-file (left side of warped image), col 7 = a-file (right side)
#   row 0 = rank 1 (top of warped image),       row 7 = rank 8 (bottom)
FILES = list("hgfedcba")
RANKS = list("12345678")

N_FRAMES   = 5       # was 8 — fewer frames, faster capture
N_FLUSH    = 6       # was 10
JPEG_Q     = 95      # restored — 88 was losing piece-shape detail
CACHE_TTL  = "1h"    # baseline + static prefix stay cached across moves


def capture_frames(camera: int, n_frames: int = N_FRAMES, n_flush: int = N_FLUSH) -> list:
    with open("board_transform.json") as f:
        d = json.load(f)
    M    = np.array(d["M"], np.float32)
    size = tuple(d["size"])

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        sys.exit(1)
    for _ in range(n_flush):
        cap.read()
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.warpPerspective(frame, M, size))
    cap.release()
    if not frames:
        print("Failed to capture frames")
        sys.exit(1)
    return frames


def median_image(frames: list) -> np.ndarray:
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def label_image(image: np.ndarray, padding: int = 36) -> np.ndarray:
    h, w = image.shape[:2]
    cell = w // 8

    canvas = np.full((h + 2*padding, w + 2*padding, 3), 245, dtype=np.uint8)
    canvas[padding:padding+h, padding:padding+w] = image

    for i, f in enumerate("hgfedcba"):
        x = padding + i*cell + cell//2 - 6
        cv2.putText(canvas, f, (x, padding - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f, (x, padding + h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

    for i, r in enumerate("12345678"):
        y = padding + i*cell + cell//2 + 6
        cv2.putText(canvas, r, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, r, (padding + w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

    for i in range(9):
        cv2.line(canvas,
                 (padding + i*cell, padding),
                 (padding + i*cell, padding + h),
                 (180, 180, 180), 1)
        cv2.line(canvas,
                 (padding, padding + i*cell),
                 (padding + w, padding + i*cell),
                 (180, 180, 180), 1)

    return canvas


_FILE_FLIP = {"a":"h","b":"g","c":"f","d":"e","e":"d","f":"c","g":"b","h":"a"}


def rotate_180(board_state: dict) -> dict:
    return {
        f"{_FILE_FLIP[sq[0]]}{9 - int(sq[1])}": piece
        for sq, piece in board_state.items()
    }


_PIECE_TO_FEN = {
    "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K",
    "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k",
}


def board_state_to_fen(board_state: dict, side_to_move: str) -> str:
    rows = []
    for rank in range(8, 0, -1):
        empty, row = 0, ""
        for f in "abcdefgh":
            piece = board_state.get(f"{f}{rank}")
            if piece and piece in _PIECE_TO_FEN:
                if empty:
                    row += str(empty)
                    empty = 0
                row += _PIECE_TO_FEN[piece]
            else:
                empty += 1
        if empty:
            row += str(empty)
        rows.append(row)
    return f"{'/'.join(rows)} {side_to_move} - - 0 1"


def best_move(board_state: dict, side: str, time_limit: float = 1.0) -> str:
    import shutil
    try:
        import chess
        import chess.engine
    except ImportError:
        raise RuntimeError("python-chess not installed. Run: pip install chess")

    sf = shutil.which("stockfish")
    if sf is None:
        raise RuntimeError(
            "Stockfish binary not found in PATH.\n"
            "Install it: sudo apt install stockfish"
        )

    fen   = board_state_to_fen(board_state, "w" if side == "white" else "b")
    board = chess.Board(fen)
    if not board.is_valid():
        raise RuntimeError(f"Detected position is not a legal chess position.\nFEN: {fen}")

    with chess.engine.SimpleEngine.popen_uci(sf) as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
    return board.san(result.move)


def draw_overlay(warped: np.ndarray, board_state: dict) -> np.ndarray:
    cell    = warped.shape[1] // 8
    display = warped.copy()
    for sq, piece in board_state.items():
        if not piece:
            continue
        file_idx = ord(sq[0]) - ord('a')
        rank     = int(sq[1])
        c = 7 - file_idx
        r = rank - 1
        x0, y0 = c * cell, r * cell
        color = (0, 200, 0) if piece.startswith("w") else (0, 100, 255)
        cv2.rectangle(display, (x0+2, y0+2), (x0+cell-2, y0+cell-2), color, 2)
        cv2.putText(display, piece, (x0+8, y0+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return display


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return base64.standard_b64encode(buf.tobytes()).decode()


# Static prompt that explains the baseline. Goes BEFORE the baseline image so
# it sits inside the cached prefix.
_STATIC_PROMPT_HEADER = (
    "You see two top-down photos of the same chess board.\n\n"
    "IMAGE 1 — BASELINE: every piece is at its standard starting square. "
    "Use this image to learn TWO things about each piece type:\n"
    "  (a) the sticker color, and\n"
    "  (b) the piece SHAPE/SILHOUETTE — the body of the chess piece itself "
    "(rook, knight, bishop, queen, king, pawn each have a distinctive top-down\n"
    "      outline: pawns are small round nubs; rooks are cylindrical with\n"
    "      crenellations; knights have an asymmetric horse-head profile;\n"
    "      bishops are tall and pointed; queens are taller with a crown ring;\n"
    "      kings are taller still with a cross on top).\n\n"
    "Baseline piece positions:\n"
    "  rank 1 (top row):     a1=wr  b1=wn  c1=wb  d1=wq  e1=wk  f1=wb  g1=wn  h1=wr\n"
    "  rank 2:               a2..h2 are all wp (white pawns)\n"
    "  rank 7:               a7..h7 are all bp (black pawns)\n"
    "  rank 8 (bottom row):  a8=br  b8=bn  c8=bb  d8=bq  e8=bk  f8=bb  g8=bn  h8=br\n\n"
    "CRITICAL — known same-color collision in this set:\n"
    "  The BLACK KNIGHT (bn) and BLACK ROOK (br) BOTH HAVE RED STICKERS.\n"
    "  Color alone CANNOT distinguish them. You MUST use the piece body shape:\n"
    "    - bn (knight): asymmetric horse-head silhouette — the top of the piece\n"
    "      has an elongated, curved profile that protrudes to one side (the\n"
    "      horse's snout). The outline is irregular, NOT circular.\n"
    "    - br (rook): cylindrical body with a flat castellated top (battlements).\n"
    "      From above, the outline is round/octagonal with small notches around\n"
    "      the rim. The shape is symmetric and compact.\n"
    "  In IMAGE 1 (baseline starting position), the knights are at b8 and g8\n"
    "  and the rooks are at a8 and h8 — study those four squares carefully to\n"
    "  lock in what each shape looks like in this camera setup, then apply that\n"
    "  same shape distinction to every red-sticker piece in IMAGE 2.\n\n"
    "Other piece types may or may not share colors; rely on the baseline image "
    "to learn the unique combination of color + shape for each piece type.\n\n"
    "Square coordinates are printed at the edges of both images:\n"
    "  - File labels along top/bottom: h g f e d c b a (left to right)\n"
    "  - Rank labels along left/right: 1 2 3 4 5 6 7 8 (top to bottom)\n"
    "  - So the TOP-LEFT square of the playing area is h1, TOP-RIGHT is a1,\n"
    "    BOTTOM-LEFT is h8, BOTTOM-RIGHT is a8.\n\n"
    "Here is IMAGE 1 — BASELINE (standard starting position). Memorize the "
    "sticker color AND the piece shape for every piece type."
)

# Tail prompt that comes after the current (volatile) image.
_TAIL_PROMPT = (
    "IMAGE 2 above is the current board (pieces may have moved). For each "
    "occupied square in IMAGE 2, identify the piece by combining BOTH clues:\n"
    "  1. Sticker color\n"
    "  2. Piece shape/silhouette\n"
    "\n"
    "When two piece types in the baseline share a similar sticker color, you "
    "MUST inspect the piece body shape carefully to choose between them. Do "
    "not guess on color alone in those cases — examine the silhouette of the "
    "piece in IMAGE 2, compare it to the matching pieces in IMAGE 1, and pick "
    "the one whose shape lines up. Think step by step on each ambiguous square.\n"
    "\n"
    "Rules:\n"
    "1. Only output squares that are OCCUPIED in IMAGE 2.\n"
    "2. Empty squares MUST NOT appear in the JSON.\n"
    "3. Valid piece codes (use only these): wr, wn, wb, wq, wk, wp, br, bn, bb, bq, bk, bp\n"
    "4. Resolve same-color collisions by piece shape — never default to one type when uncertain.\n"
    "\n"
    "Return ONLY a JSON object, no other text:\n"
    "  {\"square\": \"piece_code\", ...}\n"
    "Example: {\"e4\": \"wp\", \"d5\": \"bp\", \"a1\": \"wr\", \"h8\": \"br\"}"
)


def ask_claude(baseline: np.ndarray, current: np.ndarray) -> tuple[dict, object]:
    """
    Returns (board_state, usage) so the caller can verify cache hits.
    """
    labeled_base = label_image(baseline)
    labeled_curr = label_image(current)

    cv2.imwrite("/tmp/baseline_labeled.png", labeled_base)
    cv2.imwrite("/tmp/current_labeled.png",  labeled_curr)

    b64_base = img_to_b64(labeled_base)
    b64_curr = img_to_b64(labeled_curr)

    # Content layout — everything up to and including the baseline image is the
    # cached prefix. cache_control sits on the baseline image (last block of
    # the cached portion). The current image + tail prompt come after the
    # breakpoint and are processed fresh each call.
    content = [
        {"type": "text",  "text": _STATIC_PROMPT_HEADER},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_base},
            "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL},
        },
        {"type": "text",  "text": "IMAGE 2 — CURRENT BOARD (pieces may have moved):"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_curr}},
        {"type": "text",  "text": _TAIL_PROMPT},
    ]

    client  = anthropic.Anthropic()
    message = client.messages.create(
        model         = "claude-opus-4-7",
        max_tokens    = 16000,
        thinking      = {"type": "adaptive"},
        output_config = {"effort": "xhigh"},    # xhigh — best for visual reasoning on Opus 4.7
        messages      = [{"role": "user", "content": content}],
    )

    raw = ""
    for block in message.content:
        if block.type == "text":
            raw = block.text.strip()
            break
    if not raw:
        print("Claude returned no text block")
        sys.exit(1)

    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        print("Claude returned unexpected output:\n" + raw)
        sys.exit(1)

    try:
        result = json.loads(raw[start:end+1])
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            print("Claude returned unexpected output:\n" + raw)
            sys.exit(1)
        result = json.loads(m.group())

    return {sq: pc for sq, pc in result.items() if pc in VALID_CODES}, message.usage


def print_board(board_state: dict) -> None:
    print("   a    b    c    d    e    f    g    h")
    print("  " + "─" * 37)
    for row in range(8, 0, -1):
        line = f"{row} │"
        for col in range(8):
            sq    = f"{chr(ord('a') + col)}{row}"
            piece = board_state.get(sq, "")
            if not piece:
                line += "  .  "
            elif piece.startswith("w"):
                line += f" {GREEN}{piece:2s}{RESET}  "
            else:
                line += f" {BLUE}{piece:2s}{RESET}  "
        line += "│"
        print(line)
    print("  " + "─" * 37)
    counts: dict = {}
    for p in board_state.values():
        if p:
            counts[p] = counts.get(p, 0) + 1
    if counts:
        parts = []
        for p, n in sorted(counts.items()):
            color = GREEN if p.startswith("w") else BLUE
            parts.append(f"{color}{p}{RESET}×{n}")
        print("\nPieces: " + "  ".join(parts))
    else:
        print("\n(no pieces detected)")


def capture_baseline(camera: int, frames_count: int) -> np.ndarray:
    """Capture & save a fresh baseline image."""
    print(f"Capturing baseline ({frames_count} frames)...", flush=True)
    frames = capture_frames(camera, n_frames=frames_count)
    img    = median_image(frames)
    cv2.imwrite(BASELINE_FILE, img)
    cv2.imwrite("/tmp/baseline_labeled.png", label_image(img))
    print(f"Baseline saved → {BASELINE_FILE}")
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Opus chess detection (Architecture A)")
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--frames",   type=int, default=N_FRAMES,
                        help=f"Frames per capture (default {N_FRAMES})")
    parser.add_argument("--register", action="store_true",
                        help="Save current view as baseline only (board must be in starting position)")
    parser.add_argument("--fresh-baseline", action="store_true",
                        help="Capture a new baseline AND run detection in one command")
    parser.add_argument("--best", choices=["white", "black"], default=None,
                        help="Ask Stockfish for the best move for this side")
    parser.add_argument("--time", type=float, default=1.0,
                        help="Stockfish thinking time in seconds (default 1.0)")
    args = parser.parse_args()

    # --register: capture baseline only and exit
    if args.register:
        print("Make sure the board is in the STANDARD STARTING POSITION.")
        capture_baseline(args.camera, args.frames)
        print("Now move pieces and run without --register.")
        return

    # --fresh-baseline: capture new baseline first, then continue to detection
    if args.fresh_baseline:
        print("Make sure the board is in the STANDARD STARTING POSITION for this capture.")
        capture_baseline(args.camera, args.frames)
        input("Baseline captured. Set up the position you want detected, then press Enter...")

    baseline = cv2.imread(BASELINE_FILE)
    if baseline is None:
        print(f"ERROR: {BASELINE_FILE} not found.")
        print("Put the board in starting position and run: python claude_detect_A.py --register")
        sys.exit(1)

    print(f"Capturing {args.frames} frames of current position...", flush=True)
    frames = capture_frames(args.camera, n_frames=args.frames)
    image  = median_image(frames)

    print("Sending to Claude Opus (cached baseline)...", flush=True)
    raw_state, usage = ask_claude(baseline, image)
    board_state      = rotate_180(raw_state)

    print_board(board_state)
    print(f"\nRaw map: {board_state}")

    # Cache diagnostics
    cache_read   = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    fresh_input  = getattr(usage, "input_tokens", 0) or 0
    if cache_read:
        print(f"\n[cache HIT — {cache_read} tokens read from cache, {fresh_input} fresh]")
    elif cache_create:
        print(f"\n[cache WRITE — {cache_create} tokens cached for future calls, {fresh_input} fresh]")
    else:
        print(f"\n[no cache activity — {fresh_input} input tokens (prefix may be below cache minimum)]")

    overlay = draw_overlay(image, raw_state)
    cv2.imwrite("/tmp/claude_detection.png", overlay)

    if args.best:
        try:
            move = best_move(board_state, args.best, time_limit=args.time)
            color = GREEN if args.best == "white" else BLUE
            print(f"\n{color}Best move for {args.best}: {move}{RESET}")
            print(f"FEN: {board_state_to_fen(board_state, 'w' if args.best == 'white' else 'b')}")
        except Exception as e:
            print(f"\nStockfish failed: {e}")

    print("\nDebug images:")
    print("  baseline → /tmp/baseline_labeled.png")
    print("  current  → /tmp/current_labeled.png")
    print("  overlay  → /tmp/claude_detection.png")


if __name__ == "__main__":
    main()
