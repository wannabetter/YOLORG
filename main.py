from ultralytics import YOLO

if __name__ == '__main__':
    YOLO = YOLO('VOC/YOLOv8n_C2D_Gold/weights/best.pt')
    # YOLO.train(**{'cfg':'Yaml/Cfg/YOLOv8/YOLOv8n_VOC.yaml'})
    YOLO.predict(r"heatmap/images/dog0.jpg", show=True,save=True, visualize=True)
