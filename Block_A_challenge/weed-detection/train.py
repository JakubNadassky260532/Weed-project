from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == "__main__":
    rf = Roboflow(api_key="y0tDpkdT3O0BxonBdltn")
    project = rf.workspace("jakubs-workspace-hmvz5").project("weed-segmentation-edo5h")
    version = project.version(1)
    dataset = version.download("yolov11")
            
    model = YOLO("yolo11s-seg.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        patience=30,
        imgsz=960,
        batch=4,
        device=0,
        project="C:/yolo_runs",
    )