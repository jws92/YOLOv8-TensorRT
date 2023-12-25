import torch
from ultralytics import YOLO

import cv2
import numpy as np

import argparse
import os
import time


def get_bboxes(result):
    bboxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.int32).tolist()
    return bboxes


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
        
        results = get_bboxes(results)
    print("Elapsed time: %f msec" % ((time.time() - s) * 1000 / args.iter_durations))
    
    for res in results:
        x1, y1, x2, y2 = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    if args.save_res:
        save_res_name = os.path.splitext(args.pt_path.split("/")[-1])[0] + \
                        "_" + \
                        os.path.splitext(args.pt_path.split("/")[-1])[0] + "_res.jpg"
        cv2.imwrite(save_res_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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