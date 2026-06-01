import cv2
import os
import json
import numpy as np

CELL = 100
OUT = "data/train"

# Define ALL labels (must match training)
LABELS = [
    # Must match class_mapping.json ordering exactly
    "empty",
    "wb",
    "wn",
    "wp",
    "wr"
]

# Create directories
os.makedirs(OUT, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(OUT, label), exist_ok=True)

# Count existing images
counters = {}
for label in LABELS:
    folder = os.path.join(OUT, label)
    existing = [f for f in os.listdir(folder) if f.endswith(".png")]
    counters[label] = len(existing)
    print(f"{label}: {counters[label]} images")

# Load board transform
with open("board_transform.json") as f:
    d = json.load(f)
M = np.array(d["M"], np.float32)
size = tuple(d["size"])

cap = cv2.VideoCapture(0)

current_label = "wp"
selected_square = None

def on_click(event, x, y, flags, param):
    global selected_square
    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // CELL
        row = y // CELL
        if 0 <= row < 8 and 0 <= col < 8:
            selected_square = (row, col)
            print(f"Selected square: row={row}, col={col}")

cv2.namedWindow("board")
cv2.setMouseCallback("board", on_click)

print("\n" + "="*50)
print("CONTROLS:")
print("  1: wb (white bishop)")
print("  2: wn (white knight)")
print("  3: wp (white pawn)")
print("  4: wr (white rook)")
print("  5: empty")
print("  SPACE: Save selected square")
print("  E: Save ALL empty squares")
print("  S: Save all squares of current label")
print("  Q: Quit")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Apply perspective transform
    warped = cv2.warpPerspective(frame, M, size)
    display = warped.copy()
    
    # Draw grid
    for r in range(8):
        for c in range(8):
            color = (220, 220, 220) if (r + c) % 2 == 0 else (100, 100, 100)
            cv2.rectangle(display, (c*CELL, r*CELL), 
                         ((c+1)*CELL, (r+1)*CELL), color, -1)
            cv2.rectangle(display, (c*CELL, r*CELL), 
                         ((c+1)*CELL, (r+1)*CELL), (50, 50, 50), 1)
    
    # Add original board overlay
    overlay = cv2.addWeighted(warped, 0.3, display, 0.7, 0)
    
    # Highlight selected square
    if selected_square:
        r, c = selected_square
        cv2.rectangle(overlay, (c*CELL, r*CELL), 
                     ((c+1)*CELL, (r+1)*CELL), (0, 255, 0), 3)
    
    # Show current label
    cv2.putText(overlay, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("board", overlay)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        break
    
    # Label selection (match LABELS order)
    key_to_label = {
        ord('1'): "wb",
        ord('2'): "wn",
        ord('3'): "wp",
        ord('4'): "wr",
        ord('5'): "empty"
    }
    
    if k in key_to_label:
        current_label = key_to_label[k]
        print(f"Selected label: {current_label}")
    
    # Save single square
    if k == ord(' '):
        if selected_square is None:
            print("ERROR: No square selected! Click on a square first.")
            continue
        
        r, c = selected_square
        # Crop with padding
        pad = int(CELL * 0.1)
        crop = warped[
            max(0, r*CELL + pad):min(8*CELL, (r+1)*CELL - pad),
            max(0, c*CELL + pad):min(8*CELL, (c+1)*CELL - pad)
        ]
        
        if crop.size == 0:
            print("ERROR: Invalid crop!")
            continue
        
        # Resize to consistent size (match training / inference)
        crop = cv2.resize(crop, (100, 100))
        
        idx = counters[current_label]
        path = f"{OUT}/{current_label}/{current_label}_{idx:04d}.png"
        cv2.imwrite(path, crop)
        counters[current_label] += 1
        print(f"Saved: {path}")
    
    # Save all empty squares
    if k == ord('e'):
        print("Saving empty squares...")
        for r in range(8):
            for c in range(8):
                pad = int(CELL * 0.1)
                crop = warped[
                    max(0, r*CELL + pad):min(8*CELL, (r+1)*CELL - pad),
                    max(0, c*CELL + pad):min(8*CELL, (c+1)*CELL - pad)
                ]
                
                if crop.size == 0:
                    continue
                
                crop = cv2.resize(crop, (100, 100))
                idx = counters["empty"]
                path = f"{OUT}/empty/empty_{idx:04d}.png"
                cv2.imwrite(path, crop)
                counters["empty"] += 1
        print(f"Saved {64} empty squares")
    
    # Save all squares of current label
    if k == ord('s'):
        if current_label == "empty":
            print("Use 'E' for empty squares")
            continue
        
        print(f"Saving all squares with label {current_label}...")
        for r in range(8):
            for c in range(8):
                pad = int(CELL * 0.1)
                crop = warped[
                    max(0, r*CELL + pad):min(8*CELL, (r+1)*CELL - pad),
                    max(0, c*CELL + pad):min(8*CELL, (c+1)*CELL - pad)
                ]
                
                if crop.size == 0:
                    continue
                
                crop = cv2.resize(crop, (100, 100))
                idx = counters[current_label]
                path = f"{OUT}/{current_label}/{current_label}_{idx:04d}.png"
                cv2.imwrite(path, crop)
                counters[current_label] += 1
        print(f"Saved 64 {current_label} squares")

cap.release()
cv2.destroyAllWindows()