from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:/yolo_runs/train-4/weights/last.pt")
    model.train(resume=True)