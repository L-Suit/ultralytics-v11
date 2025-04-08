import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model_t = YOLO(r'yolo11l.pt')  # 此处填写教师模型的权重文件地址

    model_t.model.model[-1].set_Distillation = True  # 不用理会此处用于设置模型蒸馏

    model_s = YOLO("./cfg/models/11/yolo11-ADown+WTConv.yaml")  # 学生文件的yaml文件 or 权重文件地址

    imgsz = 544
    epoch = 200
    batch = 8
    optimizer = 'AdamW'
    lr0 = 0.001
    patience = 0
    weight_decay = 0.0005
    workers = 0

    model_s.train(data=r'mydataset-for31.yaml',
                  imgsz=imgsz,
                  epochs=epoch,
                  lr0=lr0,
                  batch=batch,
                  optimizer=optimizer,  # 优化器设置
                  workers=workers,
                  patience=patience,
                  # dropout=dropout,
                  weight_decay=weight_decay,

                  mosaic=0,
                  pretrained=True,
                  single_cls=False,  # 是否是单类别检测
                  close_mosaic=0,
                  device='0',
                  cache=True,
                  resume=False,  # 如过想续训,此处设置true，model不用.yaml改为last.pt的位置
                  amp=True,  # 如果出现训练损失为Nan可以关闭amp
                  # half=True,
                  # project='runs/detect',
                  project='/root/autodl-tmp/detect',
                  name=f'yolo11n-zhengliu-ADown+WTConv_for31V2_epo{epoch}_lr{lr0}_{batch}_{optimizer}_wd{weight_decay}_sz{imgsz}_',
                  model_t=model_t.model
                  )
