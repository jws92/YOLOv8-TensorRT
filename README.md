# TensorRT Conversion for YOLOv8
Simple *Python* implementation of conversion from PyTorch models to TensorRT engines for **YOLOv8**.

## Simple Summary
To convert PyTorch models to TensorRT engines, we will follow some procedures below:   
* **PyTorch** to **ONNX**
* **ONNX** to **TensorRT**
* We support **all of the tasks** of YOLOv8 models inclduing `N`, `S`, `M`, `L`, and `X`.
* We can easily convert models to the optimized engines with **FP16** or **INT8**, by using some codes in `src/`.

## Preparation
* Docker image included in this repo
* COCO validation set `val2017` to use calibration for **INT8 conversion**, as an example.
  ```bash
  wget http://images.cocodataset.org/zips/val2017.zip
  ```
* All of the model weights of YOLOv8 can be downloaded at [YOLOv8 official repo](https://github.com/ultralytics/ultralytics?tab=readme-ov-file).

## Experimental Environments
* OS - Ubuntu 20.04 (of WSL 2 in Windows 11)
* CPU - AMD Ryzen 5 5600X
* GPU - RTX 3060 12GB
* Other dependencies are following the `Dockerfile`:
  * TensorRT - 8.2.5.1
  * PyTorch - 1.12.0a0+8a1a93a
  * OpenCV - 4.5.5
  * PyCUDA - 2022.2.2

## PyTorch To ONNX
* We first convert the PyTorch models to ONNX graphs with a simple code provided by [YOLOv8](https://github.com/ultralytics/ultralytics)
```bash
# base
python export_onnx.py --weight_path /PATH/TO/WEIGHT --imgsz 640

# example: yolov8n
python export_onnx.py --weight_path ./weights/yolov8n.pt --imgsz 640

# ==> out: ./weight/yolov8n.onnx
```

## ONNX To TensorRT
* With the converted ONNX graphs, we can also convert the graphs to TensorRT engines with **FP16** or **INT8**, following the codes:
  * FP16
  ```bash
  # base
  python onnx2trt.py --onnx_path /PATH/TO/ONNX \
                     --trt_path /PATH/TO/ENGINE \
                     --fp16

  # example: yolov8n.onnx
  python onnx2trt.py --onnx_path ./weights/yolov8n.onnx \
                     --trt_path ./weights/yolov8n_fp16.engine \
                     --fp16

  # ==> out: ./weight/yolov8n_fp16.engine
  ```

  * INT8
  ```bash
  # base
  python onnx2trt.py --onnx_path /PATH/TO/ONNX \
                     --trt_path /PATH/TO/ENGINE \
                     --int8 \
                     --calib_img_path /PATH/TO/IMG_DATA_PATH \
                     --calib_cache /PATH/TO/CALIB_CACHE \
                     --calib_batch_size /PATH/TO/CALIB_BATCH_SIZE \
                     --workspace 4

  # example: yolov8n.onnx,
  # calibration dataset: COCO - val2017
  python onnx2trt.py --onnx_path ./weights/yolov8n.onnx \
                     --trt_path ./weights/yolov8n_int8.engine \
                     --int8 \
                     --calib_img_path ./val2017 \
                     --calib_cache ./caches/calib.cache \
                     --calib_batch_size 8 \
                     --workspace 10

  # ==> out: ./caches/calib.cache
  #          ./weight/yolov8n_int8.engine
  ```

## Analysis
* Elapsed time for each engine (unit: ms)
  * We measured the average elapsed time for 50 iterations on an input image (here, `bus.jpg`).
  * Although one of the fastest is INT8, it could not have dramatic difference with FP16.
    * It could be considered that some latency bottleneck may occur due to using NMS with native PyTorch, not TensorRT plugins.

* ***Detection***

|Frameworks|          N|             S|             M|              L|              X|
|:---------|         -:|            -:|            -:|             -:|             -:|
|PyTorch|     12.36 (-)|     14.43 (-)|     13.99 (-)|      19.67 (-)|      21.54 (-)|
|TRT-FP16|5.39 (+56.4%)| 6.35 (+56.0%)| 9.56 (+31.9%)| 10.97 (+44.2%)| 14.33 (+33.5%)|
|TRT-INT8|4.83 (+10.4%)| 4.95 (+22.0%)|  6.35 (+33.6%)|  9.34 (+14.9%)| 8.97 (+37.4%)|

* ***Pose***

|Frameworks|          N|             S|             M|              L|              X|
|:---------|         -:|            -:|            -:|             -:|             -:|
|PyTorch|     14.39 (-)|     14.35 (-)|     17.95 (-)|      19.53 (-)|      22.97 (-)|
|TRT-FP16|5.33 (+63.0%)| 6.33 (+55.9%)| 10.81 (+39.8%)| 11.24 (+42.4%)| 14.35 (+37.5%)|
|TRT-INT8|5.83 (-8.6%)| 5.54 (+12.5%)|  7.13 (+34.0%)|  7.52 (+33.1%)| 10.29 (+28.3%)|

* ***Segmentation***

|Frameworks|          N|             S|             M|              L|              X|
|:---------|         -:|            -:|            -:|             -:|             -:|
|PyTorch|     16.51 (-)|     17.96 (-)|     19.50 (-)|      22.35 (-)|      32.02 (-)|
|TRT-FP16|7.67 (+53.5%)| 8.42 (+53.1%)| 11.59 (+40.6%)| 14.40 (+35.6%)| 18.56 (+42.0%)|
|TRT-INT8|7.01 (+8.6%)| 8.00 (+5.0%)|  10.14 (+12.5%)|  10.68 (+25.8%)| 14.96 (+19.4%)|

* Qualitative analysis
  * IoU and confidence thresholds were set to 0.5, respectively.
  * We only analyzed the result of each engine for `bus.jpg`, and we confirmed that native PyTorch and FP16 engine was almost the same result. 
  * The results of INT8 engine had a false negative in the `bus.jpg` compared to others, but its performance was also good.

## TODO
- [X] Added classes' names on the result image
- [X] Added pose modules of both PyTorch and TRT
- [X] Added segmentation modules of both PyTorch and TRT
- [X] Added result images from all of the models

## References
* Convert ONNX to TensorRT - [LINK](https://github.com/qbxlvnf11/convert-pytorch-onnx-tensorrt/blob/TensorRT-21.08/convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py)   
* Image batcher for INT8 calibration - [LINK](https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/python/efficientdet/image_batcher.py)
* INT8 calibration - [LINK](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientdet/build_engine.py)
* TensorRT inference - [LINK](https://github.com/NVIDIA/TensorRT/blob/master/samples/python/efficientdet/infer.py)
* PyCUDA GPUArray to PyTorch Tensor - [LINK](https://discuss.pytorch.org/t/how-can-i-get-access-to-the-raw-gpu-data-of-a-tensor-for-pycuda-and-how-do-i-convert-back/21394/5)
* YOLOv8 TRT Inference   
  * Tensor preprocessing - [LINK](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L113)
  * Result tensor postprocessing - [LINK](https://github.com/ultralytics/ultralytics/blob/38eaf5e29f2a269ee50de659470c91ae44bb23f8/ultralytics/models/yolo/detect/predict.py#L23)