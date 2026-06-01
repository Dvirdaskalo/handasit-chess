import cv2
import json
import numpy as np

cap = cv2.VideoCapture(0)

print("Click 4 board corners: TL, TR, BR, BL")

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x,y])
        print("Point:", x, y)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    for p in points:
        cv2.circle(frame, tuple(p), 6, (0,0,255), -1)

    cv2.imshow("frame", frame)

    if len(points) == 4:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

src = np.array(points, np.float32)
dst = np.array([
    [0,0],
    [800,0],
    [800,800],
    [0,800]
], np.float32)

M = cv2.getPerspectiveTransform(src, dst)

data = {
    "M": M.tolist(),
    "size": [800,800]
}

with open("board_transform.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved board_transform.json")

cap.release()
cv2.destroyAllWindows()
