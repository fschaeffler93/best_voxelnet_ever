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
parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                        help='set the flag to True if dumping visualizations')
args = parser.parse_args()


dataset_dir = cfg.DATA_DIR
test_dir = os.path.join(dataset_dir, 'testing')
save_model_dir = os.path.join('.', 'save_model', args.tag)
    

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

        # just one run to initialize all the variables
        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            # param init/restore
            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(
                    sess, tf.train.latest_checkpoint(save_model_dir))
            
            
            for batch in iterate_data(test_dir, shuffle=False, aug=False, is_testset=True, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):
                tags, results = model.predict_step(sess, batch, summary=False, vis=False)
                break

            model.save_frozen_graph(sess, save_model_dir+"/frozen.pb")


if __name__ == '__main__':
    tf.app.run(main)
