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



parser = argparse.ArgumentParser(description='testing')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='pre_trained_car',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=1,
                    help='set batch size for each gpu')
parser.add_argument('-o', '--output-path', type=str, nargs='?',
                    default='./predictions', help='results output dir')
parser.add_argument('-v', '--vis', type=bool, nargs='?', default=True,
                        help='set the flag to True if dumping visualizations')
args = parser.parse_args()


dataset_dir = cfg.DATA_DIR
test_dir = os.path.join(dataset_dir, 'testing')
save_model_dir = os.path.join('.', 'save_model', args.tag)
    
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, 'data'), exist_ok=True)
if args.vis:
    os.makedirs(os.path.join(args.output_path, 'vis'), exist_ok=True)


def load_graph(frozen_graph_filename):
    """
    @param frozen_graph_filename: location of the .pb file of frozen graph
    @return: tensorflow graph definition
    """ 
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph


def main(_):
    
    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                            visible_device_list=cfg.GPU_AVAILABLE,
                            allow_growth=True)

        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )
        calib_graph = load_graph(save_model_dir + "/frozen.pb")
        with tf.Session(config=config, graph=calib_graph) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            
            writer = tf.summary.FileWriter('logs', sess.graph)
            writer.close()
        


if __name__ == '__main__':
    tf.app.run(main)
