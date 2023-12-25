import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from ultralytics.data.augment import LetterBox

import cv2
import numpy as np

import os, sys, random


# referred to the following repo:
# https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/python/efficientdet/image_batcher.py
class ImageBatcher:
    def __init__(self, input, shape, dtype, 
                 max_num_images=None, exact_batches=False, shuffle_files=False):
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
            if shuffle_files:
                random.seed(47)
                random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0 : self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

    # codes from YOLOv8
    # preprocess:
    # https://github.com/ultralytics/ultralytics/blob/38eaf5e29f2a269ee50de659470c91ae44bb23f8/ultralytics/engine/predictor.py#L113
    def preprocess_image(self, image_path):
        
        def preprocess(im, imgsz=640, half=False):
            def pre_transform(im, imgsz=640):
                letterbox = LetterBox(imgsz)
                return [letterbox(image=x) for x in im]
            
            im = pre_transform(im, imgsz)[0]
            im = im[..., ::-1].transpose((2, 0, 1))
            im = np.ascontiguousarray(im, 
                                    dtype=np.float32 if not half else np.float16) / 255.0
            
            return im
        
        img = cv2.imread(image_path)
        img = preprocess([img])
        
        return img

    def get_batch(self):
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = self.preprocess_image(image)
                
            self.batch_index += 1
            
            yield batch_data, batch_images
            
            
# referred to the following repo:
# https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientdet/build_engine.py
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file):
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            print("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, batch)
            return [int(self.batch_allocation)]
        except StopIteration:
            print("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            print("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)