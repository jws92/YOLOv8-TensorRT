import torch
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.gpuarray import GPUArray

from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox

import cv2
import numpy as np

import argparse
import os
import time

import warnings
warnings.filterwarnings('ignore')


class YOLOv8TRT:
    def __init__(self, engine_path,
                 conf=0.5, iou=0.5,
                 imgsz=640, max_det=200,
                 num_classes=80):
        self.engine_path = engine_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.num_classes = num_classes
        self.fp16 = False
        
        self.device = torch.device('cuda')
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        self._init_engine()
        
    def _init_engine(self):
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(self.engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            
            is_input = False
            if engine.binding_is_input(i):
                is_input = True
                if dtype == np.float16:
                    self.fp16 = True
                
            shape = self.context.get_binding_shape(i)
            
            if is_input and shape[0] < 0:
                profile_shape = engine.get_profile_shape(0, name)
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
                
            if is_input:
                self.batch_size = shape[0]
                
            size = dtype.itemsize
            for s in shape:
                size *= s
                
            allocation = cuda.mem_alloc(size)
            host_allocation = None
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': list(shape),
                'allocation': allocation,
                'host_allocation': host_allocation
            }
            
            self.allocations.append(allocation)
            if is_input: self.inputs.append(binding)
            else: self.outputs.append(binding)
            
    def pre_transform(self, im):
        letterbox = LetterBox(self.imgsz)
        return [letterbox(image=x) for x in im]
            
    def preprocess(self, im):
        im = np.stack(self.pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
            
        im = torch.from_numpy(im)
        im = im.to(self.device).to(torch.float16 if self.fp16 else torch.float32) / 255.0
        im = im.cpu().numpy()
            
        return im
            
    def predict(self, img):
        img_batch = self.preprocess([img])
            
        cuda.memcpy_htod(self.inputs[0]['allocation'], img_batch)
        self.context.execute_v2(self.allocations)
        res_gpuarray = GPUArray(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'],
                                gpudata=self.outputs[0]['allocation'])
                                    
        res_tensor = torch.tensor(res_gpuarray.get(),
                                  dtype=torch.float16 if self.fp16 else torch.float32,
                                  device=self.device)
        
        bboxes = self.postprocess(res_tensor, img_batch, [img])
        bboxes = self.get_bboxes(bboxes)
            
        return bboxes
            
    # postprocess
    # https://github.com/ultralytics/ultralytics/blob/38eaf5e29f2a269ee50de659470c91ae44bb23f8/ultralytics/models/yolo/detect/predict.py#L23
    def postprocess(self, preds, img, origin_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=False,
                                        max_det=self.max_det,
                                        nc=self.num_classes)
            
        if not isinstance(origin_imgs, list):
            origin_imgs = ops.convert_torch2numpy_batch(origin_imgs)
                
        bboxes = []
            
        for i, pred in enumerate(preds):
            origin_img = origin_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], origin_img.shape)
            bboxes.append(pred[:, :4])
                
        return bboxes
            
    def get_bboxes(self, bboxes):
        bboxes = bboxes[0].detach().cpu().numpy().astype(np.int32).tolist()
        return bboxes
            
            
def main(args):
    if not os.path.exists(args.trt_path) or \
        not os.path.exists(args.img_path):
        print("There are no files:")
        if not os.path.exists(args.trt_path):
            print(f"  -> {args.trt_path}")
        if not os.path.exists(args.img_path):
            print(f"  -> {args.img_path}")
        return
    
    yolo_properties = {
        'engine_path': args.trt_path,
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'max_det': 300
    }
    
    yolo = YOLOv8TRT(**yolo_properties)
    
    yolo_dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    
    for _ in range(args.warmup_durations):
        yolo.predict(yolo_dummy)
        
    img = cv2.imread(args.img_path)
        
    s = time.time()
    for _ in range(args.iter_durations):
        results = yolo.predict(img)
    print("Elapsed time: %f msec" % ((time.time() - s) * 1000 / args.iter_durations))
    
    for res in results:
        x1, y1, x2, y2 = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    if args.save_res:
        save_res_name = os.path.splitext(args.trt_path.split("/")[-1])[0] + \
                        "_" + \
                        os.path.splitext(args.trt_path.split("/")[-1])[0] + "_res.jpg"
        cv2.imwrite(save_res_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt_path", type=str, default="", help="set TRT engine path.")
    parser.add_argument("--img_path", type=str, default="", help="set image path.")
    parser.add_argument("--imgsz", type=int, default=640, help="set input image size.")
    parser.add_argument("--iou", type=float, default=0.5, help="set IoU threshold.")
    parser.add_argument("--conf", type=float, default=0.5, help="set confidence threshold.")
    parser.add_argument("--warmup_durations", type=int, default=30, 
                        help="set durations of warm-up for model to reduce bottleneck of first inference.")
    parser.add_argument("--iter_durations", type=int, default=50, 
                        help="set number of iterations of model inference.")
    parser.add_argument("--save_res", action='store_true', 
                        help="set flag for saving results.")
    
    args = parser.parse_args()
    
    main(args)
