import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import torch.backends.cudnn as cudnn
import sys
sys.path
sys.path.append('/media/ngocthien/DATA/DO_AN_TOT_NGHIEP/yolov7')

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.experimental import attempt_load


class yolov7_pose:
    def __init__(self,weights):
        self.weights = weights
        self.conf_thres = 0.01
        self.iou_thres = 0.45
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
    
    def load_model(self):
        # model = torch.load(self.weights)['model']
        cudnn.benchmark = True
        model = attempt_load(self.weights, map_location=self.device)
        model = model.half().to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self,image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = letterbox(image.copy(), stride=64, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()   # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self,image):


        img = self.preprocess_image(image)
        with torch.no_grad():
            output, _ = self.model(img)
        t1 = time.time()
        output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        print("process:",time.time() - t1)

        output = output_to_keypoint(output)

        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        
        
        if output.shape[0] >0:
            idx = np.argmax(output,axis=0)[6]
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
            cv2.rectangle(nimg,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),thickness=1,lineType=cv2.LINE_AA)
            return nimg,output[idx,7:]
        else:
            return nimg,np.array([])
    def predict1(self,image):
        img = self.preprocess_image(image)
        with torch.no_grad():
            output, _ = self.model(img)
        
        output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
            cv2.rectangle(nimg,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),thickness=1,lineType=cv2.LINE_AA)

        if output.shape[0] >0:
            output = output[:,1:]
        else:
            output = np.array([])
        return nimg,output

if __name__ == '__main__':
    model = yolov7_pose('./weight/yolov7-w6-pose.pt')
    image = cv2.imread('img.jpg')
    image,keypoints = model.predict(image)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()