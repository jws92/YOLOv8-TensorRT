import tensorrt as trt
from calibrator import Calibrator, ImageBatcher

import os
import argparse


def convert_trt_engine(args):
    if not os.path.exists(args.onnx_path):
        print(f"There is no file of {args.onnx_path}")
                
    print("Start converting ONNX to TRT engine!")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    
    if int(trt.__version__.split('.')[1]) >= 4:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)
    else:
        config.max_workspace_size = args.workspace << 30  # Unit: GB
    
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx_path, 'rb') as onnx_model:
        if not parser.parse(onnx_model.read()):
            raise RuntimeError(f"Failed to parsing {args.onnx_path} file!")
        print("Successed parsing onnx file!")
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    # set flag for dynamic shape
    dynamic = True
    if args.min_batch_size is None or \
       args.opt_batch_size is None or \
       args.max_batch_size is None:
       dynamic = False
    
    # set input shape
    for _input in inputs:
        shape = _input.shape
        if dynamic:
            profile.set_shape(
                _input.name,
                (args.min_batch_size, *shape[1:]), # min input size
                (args.opt_batch_size, *shape[1:]), # optimal input size
                (args.max_batch_size, *shape[1:])) # max input size
        else:
            # batch size = 1
            profile.set_shape(
                _input.name,
                (1, *shape[1:]),
                (1, *shape[1:]),
                (1, *shape[1:]))
    
    # set output shape
    for _output in outputs:
        shape = _output.shape
        if dynamic:
            profile.set_shape(
                _output.name,
                (args.min_batch_size, *shape[1:]), # min output size
                (args.opt_batch_size, *shape[1:]), # optimal output size
                (args.max_batch_size, *shape[1:])) # max output size
        else:
            # batch size = 1
            profile.set_shape(
                _output.name,
                (1, *shape[1:]),
                (1, *shape[1:]),
                (1, *shape[1:]))
    
    config.add_optimization_profile(profile)
    
    fp16 = args.fp16 
    int8 = args.int8
    
    # ensure run conversion
    if fp16 == int8:
        print("Set FP16 optimization.")
        fp16 = True
        int8 = False
    
    # FP16 flag
    if fp16:
        print("Set FP16 optimization.")
        config.set_flag(trt.BuilderFlag.FP16)
        
    # INT8 flag
    elif int8:
        print("Set INT8 optimization.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(args.calib_cache)
        
        # generate calibration cache file if not exists
        if args.calib_cache is None or not os.path.exists(args.calib_cache):
            calib_shape = [args.calib_batch_size] + list(inputs[0].shape[1:])
            calib_dtype = trt.nptype(inputs[0].dtype)
            config.int8_calibrator.set_image_batcher(
                ImageBatcher(args.calib_img_path, calib_shape, calib_dtype, 
                             max_num_images=args.calib_num_images,
                             exact_batches=True, shuffle_files=True))
    
    # write engine
    print("\nConverting ONNX to TRT engine...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("Failed to building TRT engine!")
    else:
        with open(args.trt_path, 'wb') as f_engine:
            f_engine.write(engine)
        print("Successed converting ONNX to TRT engine!!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="", help="set ONNX path.")
    parser.add_argument("--trt_path", type=str, default="", help="set TRT engine path.")
    parser.add_argument("--fp16", action='store_true', help="set optimization data type of \'FP16\'.")
    parser.add_argument("--int8", action='store_true', help="set optimization data type of \'INT8\'.")
    parser.add_argument("--calib_img_path", type=str, default="", help="set image data path for generating calibration cache.")
    parser.add_argument("--calib_cache", type=str, default="", help="set path of calibration cache.")
    parser.add_argument("--calib_batch_size", type=int, default=1, help="set batch size for calibration cache.")
    parser.add_argument("--calib_num_images", type=int, default=5000, help="set total number of images to use generating calibration cache.")
    parser.add_argument("--min_batch_size", type=int, default=None, help="set minimum batch size.")
    parser.add_argument("--opt_batch_size", type=int, default=None, help="set optimum batch size.")
    parser.add_argument("--max_batch_size", type=int, default=None, help="set maximum batch size.")
    parser.add_argument("--workspace", type=int, default=4, help="set workspace size (unit: GB).")
    
    args = parser.parse_args()
    
    convert_trt_engine(args)


if __name__ == '__main__':
    main()
