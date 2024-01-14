import torch
from ultralytics import YOLO

import cv2
import numpy as np

import argparse
import os, random

import time


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


def get_results(results, task='detect'):
    classes = results.boxes.cls.detach().cpu().numpy().astype(np.int32).tolist()
    bboxes = results.boxes.xyxy.detach().cpu().numpy().astype(np.int32).tolist()
    
    if task == 'detect':
        kpts = None
        vis = None
        segments = None
    
    elif task == 'pose':
        kpt_attrs = results.keypoints.data.detach().cpu().numpy()
        kpts = kpt_attrs[..., :2].astype(np.int32).tolist()
        vis = np.where(kpt_attrs[..., 2] < 0.1, 0,
                       np.where(kpt_attrs[..., 2] >= 0.9, 2, 1)).tolist()
        segments = None
    
    elif task == 'segment':
        segments = results.masks.xy
        kpts = None
        vis = None
        
    else:
        print(f"Task is not correctly selected. {task}.\n-> Default task: \'detect\'")
        kpts = None
        vis = None
        segments = None
        
    result_dict = {'classes': classes, 
                   'bboxes': bboxes, 
                   'kpts': kpts, 'vis': vis, 
                   'segments': segments}
        
    return result_dict


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


@torch.inference_mode()
def main(args):
    if not os.path.exists(args.pt_path) or \
        not os.path.exists(args.img_path):
        print("There are no files:")
        if not os.path.exists(args.pt_path):
            print(f"  -> {args.pt_path}")
        if not os.path.exists(args.img_path):
            print(f"  -> {args.img_path}")
        return
    
    # Load a pretrained weights
    model = YOLO(args.pt_path)
    model.fuse()

    for _ in range(args.warmup_durations):
        model(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8), 
              imgsz=args.imgsz,
              half=True, verbose=False)

    # Load an image
    img = cv2.imread(args.img_path)

    s = time.time()
    for _ in range(args.iter_durations):
        results = model(img, iou=args.iou, conf=args.conf,
                        imgsz=args.imgsz,
                        half=True, max_det=300,
                        verbose=False)[0]
        torch.cuda.synchronize()
        
        result_dict = get_results(results, task=args.task)
    print("Elapsed time: %f msec" % ((time.time() - s) * 1000 / args.iter_durations))
    
    img = draw_results(img, **result_dict)
    
    if args.save_res:
        save_res_name = os.path.splitext(args.pt_path.split("/")[-1])[0] + \
                        "_" + \
                        os.path.splitext(args.img_path.split("/")[-1])[0] + "_res.jpg"
        cv2.imwrite(save_res_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="detect", 
                        help="set task for TRT engine path. (default: \'detect\'; type: \'detect\', \'pose\', \'segment\')")
    parser.add_argument("--pt_path", type=str, default="", help="set .pt engine path.")
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