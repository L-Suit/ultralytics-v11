from ultralytics import YOLO, RTDETR

# Load a COCO-pretrained YOLO11n model

if __name__ == '__main__':
    model = YOLO("./cfg/models/11/yolo11-ADown+WTConv.yaml")
    # model.load('yolov8n.pt') # loading pretrain weights
    #model = RTDETR(r'./cfg/models/rt-detr/rtdetr-l.yaml')
    imgsz = 544
    epoch = 200
    batch = 8
    optimizer = 'AdamW'
    lr0 = 0.001
    patience = 0
    weight_decay = 0.0005
    workers = 0


    model.train(data=r'mydataset-for31.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                imgsz=imgsz,
                epochs=epoch,
                lr0=lr0,
                batch=batch,
                optimizer=optimizer,  # 优化器设置
                workers=workers,
                patience=patience,
                #dropout=dropout,
                weight_decay=weight_decay,

                mosaic=0,
                pretrained=True,
                single_cls=False,  # 是否是单类别检测
                close_mosaic=0,
                device='0',
                cache=True,
                resume=False, # 如过想续训,此处设置true，model不用.yaml改为last.pt的位置
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                # half=True,
                #project='runs/detect',
                project='/root/autodl-tmp/detect',
                name=f'yolo11n-test-ADown+WTConv_for31V2_epo{epoch}_lr{lr0}_{batch}_{optimizer}_wd{weight_decay}_sz{imgsz}_',
                )