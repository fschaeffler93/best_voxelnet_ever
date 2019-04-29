#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import glob
import argparse
import os
import time
import tensorflow as tf

from config import cfg
from model import RPN3D

from utils import *
from utils.kitti_loader import iterate_data, sample_test_data

from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser(description='testing')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='pre_trained_car',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=1,
                    help='set batch size for each gpu')
parser.add_argument('-o', '--output-path', type=str, nargs='?',
                    default='./predictions', help='results output dir')
parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                        help='set the flag to True if dumping visualizations')
parser.add_argument('-p', '--precision', default='FP32', help='precision for tensorRT')
args = parser.parse_args()


dataset_dir = cfg.DATA_DIR
test_dir = os.path.join(dataset_dir, 'testing')
save_model_dir = os.path.join('.', 'save_model', args.tag)
    
alloc_space_TensorRT = 6
ppgmf = (6 - alloc_space_TensorRT)/6
max_workspace_size_bytes = alloc_space_TensorRT*1000000000

def main(_):   
    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                            visible_device_list=cfg.GPU_AVAILABLE,
                            allow_growth=True)

        conf = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )

        with tf.Session(config=conf) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )        

        nd_names = model.get_output_nodes_names()
        node_list = []
        # we ned the names of the tensor, not of the ops
        for nd in nd_names:
            node_list.append(nd + ':0')

        print(node_list)
        print("\n\n\n")

        with gfile.FastGFile(save_model_dir + "/frozen.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            trt_graph = trt.create_inference_graph(input_graph_def=graph_def,outputs=node_list, 
                                                   max_batch_size=2,
                                                   max_workspace_size_bytes=max_workspace_size_bytes,
                                                   minimum_segment_size=6,
                                                   precision_mode=args.precision)
            path_new_frozen_pb = save_model_dir + "/newFrozenModel_TRT_{}.pb".format(args.precision)
            with gfile.FastGFile(path_new_frozen_pb, 'wb') as fp:
                fp.write(trt_graph.SerializeToString())
                print("TRT graph written to path ", path_new_frozen_pb)
            with tf.Session() as sess:
                writer = tf.summary.FileWriter('logs', sess.graph)
                writer.close()


if __name__ == '__main__':
    tf.app.run(main)
