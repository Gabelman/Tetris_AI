from ultralytics import YOLO

# load a custom model
model = YOLO("C:/Users/altin/OneDrive/PycharmProjects/CV/runs/runs/detect/train/weights/best.pt")

source = "C:/Users/altin/Downloads/tetris.mp4"

results = model(source, stream=True)

for result in results:
    result.show()
