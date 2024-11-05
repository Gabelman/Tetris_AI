import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

model = YOLO("C:/Users/altin/OneDrive/PycharmProjects/CV/runs/runs/detect/train/weights/best.pt")

monitor = {"top": 100, "left": 0, "width": 800, "height": 600}

# screenshots of screen are captured in real-time for processing in memory
sct = mss()

while True:
    # capture screen
    screen_img = np.array(sct.grab(monitor))

    # convert images from bgr (opencv format) to rgb (pillow format)
    img_rgb = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)

    # perform detection
    results = model(img_rgb)

    # draw results on image
    for result in results:
        for box in result.boxes:
            # get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{int(box.cls)}{box.conf:.2f}'
            cv2.rectangle(screen_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(screen_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Tetris Detection', screen_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
sct.close()
