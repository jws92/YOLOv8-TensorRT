FROM nvcr.io/nvidia/pytorch:22.05-py3

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

RUN pip install pycuda
RUN pip install ultralytics
RUN pip install opencv-python==4.5.5.64 --upgrade
RUN pip install onnx
RUN pip install onnxsim
RUN pip install onnxruntime-gpu