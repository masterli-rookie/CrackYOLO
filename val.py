from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 模型配置文件
model_yaml_path = r'../yolov11v12v13_2025_11_23/runs/V11train/exp2/weights/best.pt'
#数据集配置文件
data_yaml_path = r'../yolov11v12v13_2025_11_23/your_dataset/data.yaml' #这个就是数据集的yaml文件的路径

if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    model.val(data=data_yaml_path,
              split='val',
              imgsz=640,
              batch=4,
              # rect=False,
              project='runs/val',
              name='exp',
              save_json = True
              )