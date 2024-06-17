import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8testmodel/v8mlca.yaml')
    #model.load('yolov8n.pt') # loading pretrain weights
    model.train(data = r'/root/autodl-tmp/yolov802/datasets/nwup_10/NWPUVHR_10.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache = False,
                imgsz = 640,
                epochs = 200,
                single_cls = False,  # 是否是单类别检测
                batch = 16,
                close_mosaic = 10,
                workers = 8,
                device = '0',
                optimizer = 'SGD',  # using SGD
                #resume='runs/detect02/train25/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp = False,  # 如果出现训练损失为Nan可以关闭amp
                project = 'runs/nwpu11',
                name = 'train',
                )