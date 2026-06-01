import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import json
import os
import time
import argparse
from pathlib import Path

CELL = 100
MODEL_PATH = "model_best.pth"

# Load board transform
with open("board_transform.json") as f:
    d = json.load(f)
M = np.array(d["M"], np.float32)
size = tuple(d["size"])

# Load class mapping
with open("class_mapping.json") as f:
    class_data = json.load(f)
labels = class_data["classes"]
print(f"Loaded {len(labels)} classes: {labels}")

# Validate CELL against loaded transform size
try:
    # size is (w,h)
    inferred_cell = int(size[0] // 8)
    if inferred_cell != CELL:
        print(f"Auto-detected CELL={inferred_cell} from board_transform size; overriding default {CELL}.")
        CELL = inferred_cell
except Exception:
    pass

# Class-specific confidence thresholds (tweakable)
CLASS_THRESHOLDS = {
    'empty': 0.0,
    'wb': 0.5,
    'wn': 0.5,
    'wp': 0.68,  # require higher confidence for pawns (they're often confused)
    'wr': 0.5
}
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# Validate model output size matches labels
try:
    out_features = model.fc.out_features
    if out_features != len(labels):
        print(f"WARNING: model expects {out_features} outputs but class mapping has {len(labels)} labels.")
except Exception:
    pass

# Preprocessing (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def apply_clahe(img):
    """Apply CLAHE to the L channel in LAB space to improve contrast."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


def detect_center_color(crop):
    """Return detected dominant color in center patch: 'orange','yellow','green' or None."""
    try:
        h, w = crop.shape[:2]
        cx1 = int(w * 0.35)
        cx2 = int(w * 0.65)
        cy1 = int(h * 0.35)
        cy2 = int(h * 0.65)
        center = crop[cy1:cy2, cx1:cx2]
        if center.size == 0:
            return None
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        mean_h = int(np.mean(hsv[:, :, 0]))
        mean_s = int(np.mean(hsv[:, :, 1]))
        mean_v = int(np.mean(hsv[:, :, 2]))
        if 5 <= mean_h <= 30 and mean_s > 80 and mean_v > 50:
            return 'orange'
        if 20 <= mean_h <= 40 and mean_s > 60 and mean_v > 60:
            return 'yellow'
        if 40 <= mean_h <= 90 and mean_s > 60 and mean_v > 50:
            return 'green'
    except Exception:
        return None
    return None

cap = cv2.VideoCapture(0)

# Debugging variables
debug_square = None  # Set to (row, col) for specific square debugging
show_confidences = False

print("\nChess Piece Detection - DEBUG MODE")
print("Press 'q' to quit, 'd' to toggle debug for selected square")
print("Click on a square to select it for detailed analysis")
print("-" * 60)

selected_square = None
last_crops = {}  # store latest crop per square for manual labeling

# CLI args
parser = argparse.ArgumentParser(description='Chess piece inference with optional collection')
parser.add_argument('--collect', action='store_true', help='Enable collection mode to save uncertain crops')
parser.add_argument('--collect-dir', type=str, default='collect', help='Directory to save collected crops')
parser.add_argument('--collect-threshold', type=float, default=0.75, help='Confidence threshold for automatic collection')
parser.add_argument('--test-dir', type=str, default=None, help='Run inference on images in a directory and save CSV results')
parser.add_argument('--test-out', type=str, default='test_results.csv', help='CSV file to write test results')
args = parser.parse_args()

collect_dir = Path(args.collect_dir)
if args.collect:
    (collect_dir / 'unlabeled').mkdir(parents=True, exist_ok=True)
    (collect_dir / 'labeled').mkdir(parents=True, exist_ok=True)
    for lbl in labels:
        (collect_dir / 'labeled' / lbl).mkdir(parents=True, exist_ok=True)
    print(f"Collection enabled. Saving to: {collect_dir}")
    print("Quick-label keys:")
    for i, lbl in enumerate(labels):
        print(f"  {i+1}: {lbl}")

# If test-dir provided, run in batch test mode and exit
if args.test_dir:
    import csv
    test_path = Path(args.test_dir)
    out_csv = Path(args.test_out)
    rows = []
    img_files = sorted([p for p in test_path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
    print(f"Running test on {len(img_files)} images from {test_path}")
    for p in img_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        # optionally apply CLAHE
        img_proc = apply_clahe(img)
        input_tensor = transform(img_proc).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            lbl = labels[idx]
            conf = float(probs[idx])
        row = {'file': str(p.name), 'predicted': lbl, 'confidence': conf}
        for i, l in enumerate(labels):
            row[f'prob_{l}'] = float(probs[i])
        rows.append(row)
        print(f"{p.name}: {lbl} ({conf:.3f})")
    # write CSV
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote results to {out_csv}")
    else:
        print("No images processed.")
    raise SystemExit(0)

def mouse_callback(event, x, y, flags, param):
    global selected_square
    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // CELL
        row = y // CELL
        if 0 <= row < 8 and 0 <= col < 8:
            selected_square = (row, col)
            print(f"Selected square: {chr(ord('a')+col)}{8-row}")

cv2.namedWindow("Chess Board Detection")
cv2.setMouseCallback("Chess Board Detection", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Apply perspective transform
    warped = cv2.warpPerspective(frame, M, size)
    display = warped.copy()
    
    board_state = {}
    all_predictions = {}
    
    for r in range(8):
        for c in range(8):
            # Crop square with padding
            pad = int(CELL * 0.15)
            y1 = max(0, r * CELL + pad)
            y2 = min(8 * CELL, (r + 1) * CELL - pad)
            x1 = max(0, c * CELL + pad)
            x2 = min(8 * CELL, (c + 1) * CELL - pad)
            
            if y1 >= y2 or x1 >= x2:
                continue
            
            crop = warped[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            try:
                # Preprocess crop (contrast) and detect center color
                crop_proc = apply_clahe(crop)
                center_color = detect_center_color(crop_proc)
                # Preprocess and predict
                input_tensor = transform(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(probabilities, 1)

                    # Get all confidences and square name early
                    all_probs = probabilities[0].cpu().numpy()
                    square_name = f"{chr(ord('a') + c)}{8 - r}"

                    conf_value = float(confidence.item())
                    pred_label = labels[pred_idx.item()]

                    # Store detailed predictions
                    all_predictions[square_name] = {
                        'label': pred_label,
                        'confidence': conf_value,
                        'all_probs': {labels[i]: round(all_probs[i], 3) for i in range(len(labels))}
                    }

                    # Keep last crop for manual labeling
                    try:
                        last_crops[square_name] = crop_proc.copy()
                    except Exception:
                        pass

                    # Automatic collection conditions
                    if args.collect:
                        collect_reason = None
                        # low max confidence
                        if conf_value < args.collect_threshold:
                            collect_reason = f'low_conf_{conf_value:.3f}'
                        # wp/wr ambiguity
                        if ('wp' in labels and 'wr' in labels):
                            try:
                                if abs(wp_prob - wr_prob) < 0.12 and max(wp_prob, wr_prob) > 0.2:
                                    collect_reason = 'wp_wr_close'
                            except Exception:
                                pass
                        # color heuristic mismatch
                        if center_color == 'orange' and pred_label != 'wr':
                            collect_reason = 'color_orange_but_not_wr'

                        if collect_reason is not None:
                            ts = int(time.time() * 1000)
                            fname = f"{square_name}_{ts}_{collect_reason}.png"
                            meta = {
                                'square': square_name,
                                'predicted': pred_label,
                                'confidence': conf_value,
                                'all_probs': {labels[i]: float(all_probs[i]) for i in range(len(labels))},
                                'reason': collect_reason,
                                'timestamp': ts
                            }
                            img_path = collect_dir / 'unlabeled' / fname
                            meta_path = img_path.with_suffix('.json')
                            try:
                                cv2.imwrite(str(img_path), crop)
                                with open(meta_path, 'w') as mf:
                                    json.dump(meta, mf, indent=2)
                                print(f"Collected crop: {img_path} ({collect_reason})")
                            except Exception as e:
                                print(f"Failed to save collected crop: {e}")

                    # Pawn vs Rook disambiguation heuristic: if model predicted pawn
                    # but rook has nearly the same probability, prefer rook when
                    # its probability is reasonably high.
                    try:
                        idx_wp = labels.index('wp') if 'wp' in labels else None
                        idx_wr = labels.index('wr') if 'wr' in labels else None
                    except Exception:
                        idx_wp = idx_wr = None

                    wp_prob = all_probs[idx_wp] if idx_wp is not None else 0.0
                    wr_prob = all_probs[idx_wr] if idx_wr is not None else 0.0

                    if pred_label == 'wp' and idx_wr is not None:
                        # If rook probability is within margin of pawn and reasonably high, flip
                        margin = 0.12
                        min_wr_conf = 0.35
                        if (wr_prob + margin >= wp_prob) and (wr_prob >= min_wr_conf):
                            print(f"Heuristic: flipping {square_name} from wp({wp_prob:.3f}) to wr({wr_prob:.3f})")
                            pred_label = 'wr'
                            conf_value = float(wr_prob)

                    # If the center color looks orange, prefer rook when rook prob non-trivial
                    # If the center color suggests orange/yellow/green apply heuristics
                    if center_color is not None and idx_wr is not None:
                        if center_color in ('orange', 'yellow') and wr_prob >= 0.2:
                            print(f"Color heuristic: {square_name} looks {center_color} -> prefer wr (wr_prob={wr_prob:.3f})")
                            pred_label = 'wr'
                            conf_value = float(wr_prob)
                        # green might indicate other colored pieces, don't force label but log
                        if center_color == 'green' and pred_label not in ('wn', 'wb'):
                            print(f"Color heuristic: {square_name} looks green (label={pred_label})")

                    # Apply class-specific confidence threshold
                    threshold = CLASS_THRESHOLDS.get(pred_label, 0.6)
                    if conf_value < threshold:
                        pred_label = 'empty'
                    
            except Exception as e:
                print(f"Error processing square {r},{c}: {e}")
                pred_label = "empty"
                conf_value = 0.0
            
            # Store board state
            square = f"{chr(ord('a') + c)}{8 - r}"
            board_state[square] = pred_label
            
            # Highlight selected square
            if selected_square and selected_square == (r, c):
                cv2.rectangle(display, (c * CELL, r * CELL),
                            ((c + 1) * CELL, (r + 1) * CELL),
                            (0, 255, 255), 3)
            
            # Draw on display if not empty
            if pred_label != "empty":
                # Color code by piece type: green for white pieces (labels starting with 'w'), red for others
                color = (0, 255, 0) if pred_label.startswith('w') else (0, 0, 255)
                cv2.putText(display, pred_label, 
                           (c * CELL + 10, r * CELL + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw confidence
                cv2.putText(display, f"{conf_value:.2f} ", 
                           (c * CELL + 10, r * CELL + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw square border
            cv2.rectangle(display, (c * CELL, r * CELL),
                         ((c + 1) * CELL, (r + 1) * CELL), 
                         (255, 255, 255), 1)
            
            # Draw square label
            cv2.putText(display, f"{chr(ord('a')+c)}{8-r}", 
                       (c * CELL + 5, r * CELL + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Display
    cv2.imshow("Chess Board Detection", display)
    
    # Print detailed analysis for selected square
    if selected_square:
        r, c = selected_square
        square_name = f"{chr(ord('a') + c)}{8 - r}"
        if square_name in all_predictions:
            pred_info = all_predictions[square_name]
            print(f"\n=== Detailed analysis for {square_name} ===")
            print(f"Predicted: {pred_info['label']} (confidence: {pred_info['confidence']:.3f})")
            print("All probabilities:")
            for label, prob in sorted(pred_info['all_probs'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0.01:  # Only show probabilities > 1%
                    print(f"  {label}: {prob:.3f}")
            print("=" * 40)
    
    # Print board state
    print("\n" + "=" * 60)
    print("Board State:")
    for row in range(8, 0, -1):
        row_str = f"{row} "
        for col in range(8):
            square = f"{chr(ord('a') + col)}{row}"
            piece = board_state.get(square, "empty")
            if piece == "empty":
                row_str += " . "
            else:
                # Highlight rooks
                if piece == "wr":
                    row_str += f" \033[92m{piece}\033[0m "  # Green
                else:
                    row_str += f" {piece} "
        print(row_str)
    print("   a  b  c  d  e  f  g  h")
    
    # Count pieces
    piece_counts = {}
    for piece in board_state.values():
        if piece != "empty":
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
    
    if piece_counts:
        print("\nPiece counts:")
        for piece, count in piece_counts.items():
            print(f"  {piece}: {count}")
    print("=" * 60)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d') and selected_square:
        # Save the crop for analysis
        r, c = selected_square
        pad = int(CELL * 0.15)
        crop = warped[
            max(0, r*CELL+pad):min(8*CELL, (r+1)*CELL-pad),
            max(0, c*CELL+pad):min(8*CELL, (c+1)*CELL-pad)
        ]
        square_name = f"{chr(ord('a') + c)}{8 - r}"
        cv2.imwrite(f"debug_square_{square_name}.png", crop)
        print(f"Saved debug image: debug_square_{square_name}.png")
    elif key == ord('c'):
        # Clear selected square
        selected_square = None
        print("Cleared selection")

    # Quick manual labeling: press number keys to save selected square crop to labeled/<label>
    if args.collect and selected_square and key != 255:
        # map '1'.. to labels
        for i in range(len(labels)):
            if key == ord(str(i+1)):
                r, c = selected_square
                square_name = f"{chr(ord('a') + c)}{8 - r}"
                crop_img = last_crops.get(square_name)
                if crop_img is None:
                    print("No crop available for selected square yet.")
                    break
                label = labels[i]
                ts = int(time.time() * 1000)
                fname = f"{square_name}_{ts}_{label}.png"
                dst = collect_dir / 'labeled' / label / fname
                try:
                    cv2.imwrite(str(dst), crop_img)
                    print(f"Saved labeled crop: {dst}")
                except Exception as e:
                    print(f"Failed saving labeled crop: {e}")
                break

cap.release()
cv2.destroyAllWindows()