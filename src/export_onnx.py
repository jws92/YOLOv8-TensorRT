from ultralytics import YOLO

import os
import argparse


def export(args):
    if not os.path.exists(args.weight_path):
        print(f"There is no weight file of {args.path}")
        return
        
    model = YOLO(args.weight_path)
    model.fuse()

    model.export(format='onnx', 
                 imgsz=args.imgsz, 
                 half=True, 
                 simplify=True, 
                 workspace=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="", 
                        help="set weight path.")
    parser.add_argument("--imgsz", type=int, default=640, 
                        help="set input image size. (default: 640)")
    
    args = parser.parse_args()
    
    export(args)
