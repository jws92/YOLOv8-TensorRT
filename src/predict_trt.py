import torch
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.gpuarray import GPUArray

from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox

import cv2
import numpy as np

import argparse, os, random, time, warnings
warnings.filterwarnings('ignore')


# for detection and segmentation tasks
COCO_CLASSES = {
    0: u'person', 1: u'bicycle', 2: u'car', 3: u'motorcycle', 4: u'airplane',
    5: u'bus', 6: u'train', 7: u'truck', 8: u'boat', 9: u'traffic light', 
    10: u'fire hydrant', 11: u'stop sign', 12: u'parking meter', 13: u'bench', 14: u'bird',
    15: u'cat', 16: u'dog', 17: u'horse', 18: u'sheep', 19: u'cow',
    20: u'elephant', 21: u'bear', 22: u'zebra', 23: u'giraffe', 24: u'backpack',
    25: u'umbrella', 26: u'handbag', 27: u'tie', 28: u'suitcase', 29: u'frisbee',
    30: u'skis', 31: u'snowboard', 32: u'sports ball', 33: u'kite', 34: u'baseball bat',
    35: u'baseball glove', 36: u'skateboard', 37: u'surfboard', 38: u'tennis racket', 39: u'bottle',
    40: u'wine glass', 41: u'cup', 42: u'fork', 43: u'knife', 44: u'spoon',
    45: u'bowl', 46: u'banana', 47: u'apple', 48: u'sandwich', 49: u'orange',
    50: u'broccoli', 51: u'carrot', 52: u'hot dog', 53: u'pizza', 54: u'donut',
    55: u'cake', 56: u'chair', 57: u'couch', 58: u'potted plant', 59: u'bed',
    60: u'dining table', 61: u'toilet', 62: u'tv', 63: u'laptop', 64: u'mouse',
    65: u'remote', 66: u'keyboard', 67: u'cell phone', 68: u'microwave', 69: u'oven',
    70: u'toaster', 71: u'sink', 72: u'refrigerator', 73: u'book', 74: u'clock',
    75: u'vase', 76: u'scissors', 77: u'teddy bear', 78: u'hair drier', 79: u'toothbrush'
}


# for pose tasks
COCO_KPTS_CLASSES = {
    0: u'person'
}

COCO_KPTS_SKELETON = [
    [16,14], [14,12], [17,15], [15,13], [12,13], 
    [6,12], [7,13], [6,7], [6,8], [7,9], 
    [8,10], [9,11], [2,3], [1,2], [1,3], 
    [2,4], [3,5], [4,6], [5,7]]


class YOLOv8TRT:
    def __init__(self, engine_path, task,
                 conf=0.5, iou=0.5,
                 imgsz=640, max_det=200,
                 num_classes=80,
                 kpt_shape=None):
        self.engine_path = engine_path
        self.task = task
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.num_classes = num_classes
        self.kpt_shape = kpt_shape
        
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
            
    # engine inference
    def predict(self, img):
        img_batch = self.preprocess([img])
            
        cuda.memcpy_htod(self.inputs[0]['allocation'], img_batch)
        self.context.execute_v2(self.allocations)
        res_gpuarray = GPUArray(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'],
                                gpudata=self.outputs[0]['allocation'])
        # this would be mask tensor if inference type is `segment`
        res_tensor = torch.tensor(res_gpuarray.get(),
                                  dtype=torch.float16 if self.fp16 else torch.float32,
                                  device=self.device)
        
        if self.task == 'detect':
            result_dict = self.det_postprocess(res_tensor, img_batch, [img])
        
        elif self.task == 'pose':
            result_dict = self.pose_postprocess(res_tensor, img_batch, [img])
        
        elif self.task == 'segment':
            res_gpuarray_2 = GPUArray(self.outputs[1]['shape'], dtype=self.outputs[1]['dtype'],
                                      gpudata=self.outputs[1]['allocation'])
            # this would be bbox tensor if inference type is `segment`
            res_tensor_2 = torch.tensor(res_gpuarray_2.get(),
                                        dtype=torch.float16 if self.fp16 else torch.float32,
                                        device=self.device)
            result_dict = self.mask_postprocess([res_tensor_2, res_tensor], img_batch, [img])
        else:
            raise NotImplementedError("You can choose inference type of \'detect\', \'pose\', and \'segment\'.")
            
        return result_dict
            
    # detection - postprocess
    # https://github.com/ultralytics/ultralytics/blob/38eaf5e29f2a269ee50de659470c91ae44bb23f8/ultralytics/models/yolo/detect/predict.py#L23
    def det_postprocess(self, preds, img, origin_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=False,
                                        max_det=self.max_det,
                                        nc=self.num_classes)
        
        if not isinstance(origin_imgs, list):
            origin_imgs = ops.convert_torch2numpy_batch(origin_imgs)
        
        classes = []
        bboxes = []
        
        for i, pred in enumerate(preds):
            origin_img = origin_imgs[i]
            # get bboxes
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], origin_img.shape)
            
            # gpu to cpu
            pred = pred.detach().cpu().numpy().astype(np.int32)
            # add classes
            classes.append(pred[:, 5])
            # add bboxes
            bboxes.append(pred[:, :4])
        
        result_dict = {'classes': classes[0], 
                       'bboxes': bboxes[0], 
                       'kpts': None, 'vis': None, 
                       'segments': None}
        
        return result_dict
    
    # pose - postprocess
    # https://github.com/ultralytics/ultralytics/blob/bc5b528ca78c67a689ce6001af963aaa1a5b2d5e/ultralytics/models/yolo/pose/predict.py#L31
    def pose_postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=False,
                                        max_det=self.max_det,
                                        nc=self.num_classes)
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            
        classes = []
        bboxes = []
        kpts = []
        
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            # get bboxes
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            # get keypoints' attributes
            pred_kpts = pred[:, 6:].view(len(pred), *self.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            
            # gpu to cpu
            pred = pred.detach().cpu().numpy().astype(np.int32)
            # add classes
            classes.append(pred[:, 5])
            # add bboxes
            bboxes.append(pred[:, :4])
            # add keypoints' attributes
            kpts.append(pred_kpts.detach().cpu().numpy())
            
        # get keypoint coordinates and visibilities
        kpts, vis = self.split_kpts_vis(kpts[0])
            
        result_dict = {'classes': classes[0], 
                       'bboxes': bboxes[0], 
                       'kpts': kpts, 'vis': vis, 
                       'segments': None}
        
        return result_dict
        
    # decode keypoints and visibilities from results
    def split_kpts_vis(self, kpt_attrs):
        kpts = kpt_attrs[..., :2].astype(np.int32)
        # vis < 0.1          -->> not_labeled; 0
        # 0.1 <= vis < 0.9   -->> not_visible; 1
        # 0.9 <= vis         -->> visible; 2
        vis = np.where(kpt_attrs[..., 2] < 0.1, 0,
                       np.where(kpt_attrs[..., 2] >= 0.9, 2, 1))
        
        return kpts, vis
        
    # segmentation - postprocess
    # https://github.com/ultralytics/ultralytics/blob/bc5b528ca78c67a689ce6001af963aaa1a5b2d5e/ultralytics/models/yolo/segment/predict.py#L28
    def mask_postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(preds[0],
                                    self.conf,
                                    self.iou,
                                    agnostic=False,
                                    max_det=self.max_det,
                                    nc=self.num_classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            
        classes = []
        bboxes = []
        segments = []
            
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            if not len(pred):  # save empty boxes
                classes.append(None)
                bboxes.append(None)
                segments.append(None)
            else:
                # get masks
                mask = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                # get bboxes
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                
                # gpu to cpu
                pred = pred.detach().cpu().numpy().astype(np.int32)
                mask = mask.detach().cpu().numpy().astype(np.uint8)
                
                # add classes
                classes.append(pred[:, 5])
                # add bboxes
                bboxes.append(pred[:, :4])
                # convert mask to segments
                mask = self.masks2segments(mask)
                segment = [ops.scale_coords(img.shape[2:], m, orig_img.shape)
                           for m in mask]
                # add segments
                segments.append(segment)
        
        result_dict = {'classes': classes[0], 
                       'bboxes': bboxes[0], 
                       'kpts': None, 'vis': None, 
                       'segments': segments[0]}
        
        return result_dict
        
    # get segmentation contours from masks
    # https://github.com/ultralytics/ultralytics/blob/bc5b528ca78c67a689ce6001af963aaa1a5b2d5e/ultralytics/utils/ops.py#L748
    def masks2segments(self, masks, strategy='largest'):
        segments = []
        for x in masks:
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if c:
                if strategy == 'concat':
                    c = np.concatenate([x.reshape(-1, 2) for x in c])
                elif strategy == 'largest':
                    c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))
            segments.append(c.astype(np.float32))
        return segments
    
    
def draw_results(img, 
                 classes=None, 
                 bboxes=None, 
                 kpts=None, vis=None, 
                 segments=None):
    if bboxes is None: 
        return None
    
    draw_img = img.copy()
    
    # set random colors for each object
    colors = [
        (random.randint(0, 255), # b
         random.randint(0, 255), # g
         random.randint(0, 255)) # r
        for _ in range(len(bboxes))]
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # draw bbox
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), colors[i], 2)
        
        # draw classes' names
        cls_name = COCO_CLASSES[classes[i]]
        cv2.rectangle(draw_img, (x1, y1-25), (x1+15*len(cls_name), y1), colors[i], -1)
        cv2.putText(draw_img, "%s" % cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        if kpts is not None:
            # draw skeletons
            for s_idx in range(len(COCO_KPTS_SKELETON)):
                if vis[i][COCO_KPTS_SKELETON[s_idx][0]-1] != 0 and vis[i][COCO_KPTS_SKELETON[s_idx][1]-1] != 0:
                    if kpts[i][COCO_KPTS_SKELETON[s_idx][0]-1][0] >= x1 and kpts[i][COCO_KPTS_SKELETON[s_idx][0]-1][0] <= x2 and \
                       kpts[i][COCO_KPTS_SKELETON[s_idx][0]-1][1] >= y1 and kpts[i][COCO_KPTS_SKELETON[s_idx][0]-1][1] <= y2 and \
                       kpts[i][COCO_KPTS_SKELETON[s_idx][1]-1][0] >= x1 and kpts[i][COCO_KPTS_SKELETON[s_idx][1]-1][0] <= x2 and \
                       kpts[i][COCO_KPTS_SKELETON[s_idx][1]-1][1] >= y1 and kpts[i][COCO_KPTS_SKELETON[s_idx][1]-1][1] <= y2:
                       cv2.line(draw_img, 
                                (kpts[i][COCO_KPTS_SKELETON[s_idx][0]-1][0], kpts[i][COCO_KPTS_SKELETON[s_idx][0]-1][1]), 
                                (kpts[i][COCO_KPTS_SKELETON[s_idx][1]-1][0], kpts[i][COCO_KPTS_SKELETON[s_idx][1]-1][1]), 
                                (0, 255, 255), 2)
                        
            # draw keypoints
            for k, v in zip(kpts[i], vis[i]):
                if k[0] >= x1 and k[0] <= x2 and k[1] >= y1 and k[1] <= y2:
                    # visible
                    if v == 2: cv2.circle(draw_img, (k[0], k[1]), 3, (0, 255, 0), 3)
                    # not visible
                    elif v == 1: cv2.circle(draw_img, (k[0], k[1]), 3, (0, 0, 255), 3)
                    # not labeled
                    else: cv2.circle(draw_img, (k[0], k[1]), 3, (0, 0, 0), 3)
                
        # draw segmentations
        if segments is not None:
            cv2.drawContours(draw_img, [segments[i].astype(np.int32)], 0, colors[i], 2)
    
    return draw_img

            
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
        'task': args.task,
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'max_det': 300,
        'num_classes': len(COCO_CLASSES) if args.task != 'pose' else len(COCO_KPTS_CLASSES),
        'kpt_shape': [args.num_kpts, 3]
    }
    
    yolo = YOLOv8TRT(**yolo_properties)
    
    # warming up
    # ==================================================
    yolo_dummy = np.zeros((args.imgsz, args.imgsz, 3), 
                          dtype=np.uint8)
    for _ in range(args.warmup_durations):
        yolo.predict(yolo_dummy)
    # ==================================================
    
    img = cv2.imread(args.img_path)
        
    s = time.time()
    for _ in range(args.iter_durations):
        result_dict = yolo.predict(img)
    print("Elapsed time: %f msec" % ((time.time() - s) * 1000 / args.iter_durations))
        
    img = draw_results(img, **result_dict)
        
    if args.save_res:
        save_res_name = os.path.splitext(args.trt_path.split("/")[-1])[0] + \
                        "_" + \
                        os.path.splitext(args.img_path.split("/")[-1])[0] + "_res.jpg"
        cv2.imwrite(save_res_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="detect", 
                        help="set task for TRT engine path. (default: \'detect\'; type: \'detect\', \'pose\', \'segment\')")
    parser.add_argument("--trt_path", type=str, default="", 
                        help="set TRT engine path.")
    parser.add_argument("--img_path", type=str, default="", 
                        help="set image path.")
    parser.add_argument("--imgsz", type=int, default=640, 
                        help="set input image size. (default: 640)")
    parser.add_argument("--num_kpts", type=int, default=17, 
                        help="set number of keypoints. (default: 17)")
    parser.add_argument("--iou", type=float, default=0.5, 
                        help="set IoU threshold. (default: 0.5)")
    parser.add_argument("--conf", type=float, default=0.5, 
                        help="set confidence threshold. (default: 0.5)")
    parser.add_argument("--warmup_durations", type=int, default=30, 
                        help="set durations of warm-up for model to reduce bottleneck of first inference. (default: 30)")
    parser.add_argument("--iter_durations", type=int, default=50, 
                        help="set number of iterations of model inference. (default: 50)")
    parser.add_argument("--save_res", action='store_true', 
                        help="set flag for saving results.")
    
    args = parser.parse_args()
    
    main(args)
