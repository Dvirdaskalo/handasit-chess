#!/usr/bin/env python3
"""
Chess piece detection via Claude Opus Vision.

Workflow:
  1. Put board in standard starting position, run:
        python claude_detect.py --register
     This saves the current view as the baseline image.
  2. After moving pieces, run:
        python claude_detect.py
     The script sends BOTH images (baseline + current) to Claude Opus in one
     prompt. Claude learns each piece's sticker color from the baseline (where
     every starting square is known), then identifies the pieces in the
     current board.
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

N_FRAMES = 8


def capture_frames(camera: int, n_frames: int = N_FRAMES, n_flush: int = 10) -> list:
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
    """Per-pixel temporal median — kills camera noise and reflections."""
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def label_image(image: np.ndarray, padding: int = 36) -> np.ndarray:
    """
    Place file/rank labels around the warped board so Claude can locate
    each square unambiguously. Also draws a faint grid.
    """
    h, w = image.shape[:2]
    cell = w // 8

    canvas = np.full((h + 2*padding, w + 2*padding, 3), 245, dtype=np.uint8)
    canvas[padding:padding+h, padding:padding+w] = image

    # File labels (top + bottom)
    for i, f in enumerate("hgfedcba"):
        x = padding + i*cell + cell//2 - 6
        cv2.putText(canvas, f, (x, padding - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f, (x, padding + h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

    # Rank labels (left + right)
    for i, r in enumerate("12345678"):
        y = padding + i*cell + cell//2 + 6
        cv2.putText(canvas, r, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, r, (padding + w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

    # Faint grid
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
    """
    180° board rotation: mirror files (a↔h, b↔g, …) AND ranks (1↔8, 2↔7, …).
    The camera views the board upside-down relative to the user's perspective,
    so e.g. b2 (Claude) → g7 (user). Applied in code — Claude is never told.
    """
    return {
        f"{_FILE_FLIP[sq[0]]}{9 - int(sq[1])}": piece
        for sq, piece in board_state.items()
    }


_PIECE_TO_FEN = {
    "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K",
    "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k",
}


def board_state_to_fen(board_state: dict, side_to_move: str) -> str:
    """Convert {square: piece_code} → FEN string. side_to_move is 'w' or 'b'."""
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
    """
    Ask Stockfish for the best move for `side` ('white' or 'black').
    Returns SAN notation like 'Nf3' or 'e4'.
    Requires: pip install chess  +  stockfish binary in PATH.
    """
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
    """Draw colored rectangles + labels on each occupied square."""
    cell    = warped.shape[1] // 8
    display = warped.copy()
    for sq, piece in board_state.items():
        if not piece:
            continue
        file_idx = ord(sq[0]) - ord('a')   # 0=a … 7=h
        rank     = int(sq[1])              # 1–8
        c = 7 - file_idx                   # a→col7, h→col0
        r = rank - 1                       # rank1→row0, rank8→row7
        x0, y0 = c * cell, r * cell
        color = (0, 200, 0) if piece.startswith("w") else (0, 100, 255)
        cv2.rectangle(display, (x0+2, y0+2), (x0+cell-2, y0+cell-2), color, 2)
        cv2.putText(display, piece, (x0+8, y0+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return display


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.standard_b64encode(buf.tobytes()).decode()


def ask_claude(baseline: np.ndarray, current: np.ndarray) -> dict:
    labeled_base = label_image(baseline)
    labeled_curr = label_image(current)

    cv2.imwrite("/tmp/baseline_labeled.png", labeled_base)
    cv2.imwrite("/tmp/current_labeled.png",  labeled_curr)

    b64_base = img_to_b64(labeled_base)
    b64_curr = img_to_b64(labeled_curr)

    prompt = (
        "You see two top-down photos of the same chess board.\n\n"
        "IMAGE 1 — BASELINE: every piece is at its standard starting square. "
        "Use this image to LEARN what each piece's colored sticker looks like:\n"
        "  rank 1 (top row):     a1=wr  b1=wn  c1=wb  d1=wq  e1=wk  f1=wb  g1=wn  h1=wr\n"
        "  rank 2:               a2..h2 are all wp (white pawns)\n"
        "  rank 7:               a7..h7 are all bp (black pawns)\n"
        "  rank 8 (bottom row):  a8=br  b8=bn  c8=bb  d8=bq  e8=bk  f8=bb  g8=bn  h8=br\n\n"
        "IMAGE 2 — CURRENT: the same board after pieces have been moved. "
        "For each occupied square in IMAGE 2, match the sticker color to a piece "
        "type you learned from IMAGE 1.\n\n"
        "Square coordinates are printed at the edges of both images:\n"
        "  - File labels along top/bottom: h g f e d c b a (left to right)\n"
        "  - Rank labels along left/right: 1 2 3 4 5 6 7 8 (top to bottom)\n"
        "  - So the TOP-LEFT square of the playing area is h1, TOP-RIGHT is a1,\n"
        "    BOTTOM-LEFT is h8, BOTTOM-RIGHT is a8.\n\n"
        "Rules:\n"
        "1. Only output squares that are OCCUPIED in IMAGE 2.\n"
        "2. Empty squares MUST NOT appear in the JSON.\n"
        "3. Valid piece codes (use only these): wr, wn, wb, wq, wk, wp, br, bn, bb, bq, bk, bp\n"
        "4. Match by sticker color — ignore piece body and board square color.\n\n"
        "Return ONLY a JSON object, no other text:\n"
        "  {\"square\": \"piece_code\", ...}\n"
        "Example: {\"e4\": \"wp\", \"d5\": \"bp\", \"a1\": \"wr\", \"h8\": \"br\"}"
    )

    client  = anthropic.Anthropic()
    message = client.messages.create(
        model         = "claude-opus-4-7",
        max_tokens    = 16000,
        thinking      = {"type": "adaptive"},
        output_config = {"effort": "high"},
        messages      = [{
            "role": "user",
            "content": [
                {"type": "text",  "text": "IMAGE 1 — BASELINE (standard starting position):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_base}},
                {"type": "text",  "text": "IMAGE 2 — CURRENT BOARD (pieces may have moved):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_curr}},
                {"type": "text",  "text": prompt},
            ],
        }],
    )

    # With thinking enabled, content has ThinkingBlock(s) before the TextBlock
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

    # Keep only valid piece codes
    return {sq: pc for sq, pc in result.items() if pc in VALID_CODES}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Opus chess detection")
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--frames",   type=int, default=N_FRAMES,
                        help=f"Frames per capture (default {N_FRAMES})")
    parser.add_argument("--register", action="store_true",
                        help="Save current view as baseline (board must be in starting position)")
    parser.add_argument("--best", choices=["white", "black"], default=None,
                        help="Ask Stockfish for the best move for this side")
    parser.add_argument("--time", type=float, default=1.0,
                        help="Stockfish thinking time in seconds (default 1.0)")
    args = parser.parse_args()

    print(f"Capturing {args.frames} frames...", flush=True)
    frames = capture_frames(args.camera, n_frames=args.frames)
    image  = median_image(frames)

    if args.register:
        cv2.imwrite(BASELINE_FILE, image)
        cv2.imwrite("/tmp/baseline_labeled.png", label_image(image))
        print(f"Saved baseline to {BASELINE_FILE}")
        print("Labeled preview at /tmp/baseline_labeled.png")
        print("Now move pieces and run without --register.")
        return

    baseline = cv2.imread(BASELINE_FILE)
    if baseline is None:
        print(f"ERROR: {BASELINE_FILE} not found.")
        print("Put the board in starting position and run: python claude_detect.py --register")
        sys.exit(1)

    print("Sending baseline + current to Claude Opus...", flush=True)
    raw_state    = ask_claude(baseline, image)
    board_state  = rotate_180(raw_state)   # mirror files AND ranks for display

    print_board(board_state)
    print(f"\nRaw map: {board_state}")

    # Overlay shows what Claude actually saw, in camera coordinates
    # (no flip applied — useful for diagnosing identification errors).
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
