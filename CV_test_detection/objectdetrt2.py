import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

# SECOND APPROACH

model = YOLO("C:/Users/altin/OneDrive/PycharmProjects/CV/runs/runs/detect/train/weights/best.pt")

while True:
    # get image of screen
    screen = pyautogui.screenshot()

    screen_array = np.array(screen)

    cropped_region = screen_array[25:625, 1122:, :]

    corrected_colors = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR)

    # make detections
    results = model(corrected_colors)

    #for result in results:
    #    boxes = result.boxes

    cv2.imshow("Tetris Detection", np.squeeze(corrected_colors))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
