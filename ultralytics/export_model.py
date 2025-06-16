from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model

if __name__ == '__main__':
    model = YOLO("./cfg/models/11/yolo11-CPA+ADown+WTConv.yaml")
    model.export(format='onnx')