from ultralytics import YOLO


if __name__ == '__main__':
    YOLO = YOLO('VOC/Basketball/YOLOv8n_C2fDCN_C2f/weights/best.pt')
    YOLO.val(data='voc.yaml', batch=32, workers=0, device='0')