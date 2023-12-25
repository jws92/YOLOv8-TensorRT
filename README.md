# TensorRT Conversion for YOLOv8
Simple implementation of conversion from PyTorch models to TensorRT engines for **YOLOv8**.

## Simple Summary
To convert PyTorch models to TensorRT engines, we will follow some procedures below:   
* **PyTorch** to **ONNX**
* **ONNX** to **TensorRT**
* We target **YOLOv8-det** models inclduing `N`, `S`, `M`, `L`, and `X`.
* We can easily convert models to the optimized engines with **FP16** or **INT8**, by using some codes in `src/`.

## Preparation
* Docker image included in this repo
* COCO validation set `val2017` to use calibration for **INT8 conversion**, as an example.
  ```bash
  wget http://images.cocodataset.org/zips/val2017.zip
  ```
* All of the model weights of YOLOv8-det can be downloaded at [YOLOv8 official repo](https://github.com/ultralytics/ultralytics?tab=readme-ov-file).

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
python export_onnx.py --path /PATH/TO/WEIGHT --imgsz 640

# example: yolov8n
python export_onnx.py --path./weights/yolov8n.pt --imgsz 640

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
                     --calib_batch_size /PATH/TO/CALIB_BATCH_SIZE

  # example: yolov8n.onnx,
  # calibration dataset: COCO - val2017
  python onnx2trt.py --onnx_path ./weights/yolov8n.onnx \
                     --trt_path ./weights/yolov8n_int8.engine \
                     --int8 \
                     --calib_img_path ./val2017 \
                     --calib_cache ./caches/calib.cache \
                     --calib_batch_size 8

  # ==> out: ./caches/calib.cache
  #          ./weight/yolov8n_int8.engine
  ```

## Analysis
* Elapsed time for each engine (unit: ms)
  * We measured the average elapsed time for 50 iterations on an input image (here, `bus.jpg`).
  * Although one of the fastest is INT8, it could not have dramatic difference with FP16.
    * It could be considered that some latency bottleneck may occur due to using NMS with native PyTorch, not TensorRT plugins.

|Frameworks|          N|             S|             M|              L|              X|
|:---------|         -:|            -:|            -:|             -:|             -:|
|PyTorch|     15.81 (-)|     16.77 (-)|     15.89 (-)|      22.33 (-)|      27.22 (-)|
|TRT-FP16|6.72 (+57.5%)| 8.16 (+51.3%)| 8.48 (+46.6%)| 10.71 (+52.0%)| 14.83 (+45.5%)|
|TRT-INT8|5.70 (+15.2%)| 6.09 (+25.4%)|  7.83 (+7.7%)|  8.88 (+17.1%)| 11.61 (+21.7%)|

* Qualitative analysis
  * IoU and confidence thresholds were set to 0.5, respectively.
  * We only analyzed the result of each engine for `bus.jpg`, and we confirmed that native PyTorch and FP16 engine was almost the same result. 
  * The results of INT8 engine had a false negative in the `bus.jpg` compared to others, but its performance was also good.

|Models|PyTorch|TRT-FP16|TRT-INT8|
|:----:|:-----:|:------:|:------:|
|N|<img src=./trt_res_imgs/n/yolov8n_yolov8n_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/n/yolov8n_fp16_yolov8n_fp16_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/n/yolov8n_int8_yolov8n_int8_res.jpg width="202.5" height="207">|
|S|<img src=./trt_res_imgs/s/yolov8s_yolov8s_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/s/yolov8s_fp16_yolov8s_fp16_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/s/yolov8s_int8_yolov8s_int8_res.jpg width="202.5" height="207">|
|M|<img src=./trt_res_imgs/m/yolov8m_yolov8m_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/m/yolov8m_fp16_yolov8m_fp16_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/m/yolov8m_int8_yolov8m_int8_res.jpg width="202.5" height="207">|
|L|<img src=./trt_res_imgs/l/yolov8l_yolov8l_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/l/yolov8l_fp16_yolov8l_fp16_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/l/yolov8l_int8_yolov8l_int8_res.jpg width="202.5" height="207">|
|X|<img src=./trt_res_imgs/x/yolov8x_yolov8x_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/x/yolov8x_fp16_yolov8x_fp16_res.jpg width="202.5" height="207">|<img src=./trt_res_imgs/x/yolov8x_int8_yolov8x_int8_res.jpg width="202.5" height="207">|

## TODO
- [ ] Add classes' names and confidence value on the result image
- [ ] Add pose module

## References
* Convert ONNX to TensorRT - [LINK](https://github.com/qbxlvnf11/convert-pytorch-onnx-tensorrt/blob/TensorRT-21.08/convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py)   
* Image batcher for INT8 calibration - [LINK](https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/python/efficientdet/image_batcher.py)
* INT8 calibration - [LINK](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientdet/build_engine.py)
* TensorRT inference - [LINK](https://github.com/NVIDIA/TensorRT/blob/master/samples/python/efficientdet/infer.py)
* PyCUDA GPUArray to PyTorch Tensor - [LINK](https://discuss.pytorch.org/t/how-can-i-get-access-to-the-raw-gpu-data-of-a-tensor-for-pycuda-and-how-do-i-convert-back/21394/5)
* YOLOv8 TRT Inference   
  * Tensor preprocessing - [LINK](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py#L113)
  * Result tensor postprocessing - [LINK](https://github.com/ultralytics/ultralytics/blob/38eaf5e29f2a269ee50de659470c91ae44bb23f8/ultralytics/models/yolo/detect/predict.py#L23)