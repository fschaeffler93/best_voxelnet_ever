#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import glob
import argparse
import os
import time
import tensorflow as tf
import sys

from config import cfg
from model import RPN3D

from utils import *
from utils.kitti_loader import iterate_data, sample_test_data
from train_hook import check_if_should_pause



parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=1,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-d', '--decrease', type=bool, nargs='?', default=False,
                    help='set the flag to True if decrease model')
parser.add_argument('-m', '--minimize', type=bool, nargs='?', default=False,
                    help='set the flag to True if minimize model')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=1,
                    help='set batch size for each gpu')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.0,
                    help='set alpha in los function')
parser.add_argument('-be', '--beta', type=float, nargs='?', default=10.0,
                    help='set beta in los function')
parser.add_argument('-o', '--output-path', type=str, nargs='?', default='predictions',
                    help='results output dir')
args = parser.parse_args()


dataset_dir = cfg.DATA_DIR
train_dir = os.path.join(dataset_dir, 'training')
val_dir = os.path.join(dataset_dir, 'validation')
log_dir = os.path.join('.', 'log', args.tag)
res_dir = os.path.join('.', args.output_path)
save_model_dir = os.path.join('.', 'save_model', args.tag)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)

def main(_):
    
    with tf.Graph().as_default():
    
        start_epoch = 0
        global_counter = 0

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
            visible_device_list=cfg.GPU_AVAILABLE,
            allow_growth=True
        )

        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={"GPU": cfg.GPU_USE_COUNT,},
            allow_soft_placement=True,
        )

        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                decrease=args.decrease,
                minimize=args.minimize,
                single_batch_size=args.single_batch_size,
                learning_rate=args.lr,
                max_gradient_norm=5.0,
                alpha=args.alpha,
                beta=args.beta,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            
            # param init/restore
            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
                start_epoch = model.epoch.eval() + 1
                global_counter = model.global_step.eval() + 1
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()

            # train and validate
            is_summary, is_summary_image, is_validate = False, False, False

            summary_interval = 5
            summary_val_interval = 10
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)


            # training
            for epoch in range(start_epoch, args.max_epoch):
                counter = 0
                batch_time = time.time()
                for batch in iterate_data(train_dir, shuffle=True, aug=True, is_testset=False, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):
                    
                    counter += 1
                    global_counter += 1
                    
                    if counter % summary_interval == 0:
                        is_summary = True
                    else:
                        is_summary = False
                    
                    start_time = time.time()
                    ret = model.train_step(sess, batch, train=True, summary = is_summary)
                    forward_time = time.time() - start_time
                    batch_time = time.time() - batch_time
                    
                    print('train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f}'.format(counter, epoch + 1, args.max_epoch, ret[0], ret[1], ret[2], ret[3], ret[4], forward_time, batch_time))
                    with open(os.path.join('log', 'train.txt'), 'a') as f:
                        f.write('train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f} \n'.format(counter, epoch + 1, args.max_epoch, ret[0], ret[1], ret[2], ret[3], ret[4], forward_time, batch_time))
                    
                    if counter % summary_interval == 0:
                        print("summary_interval now")
                        summary_writer.add_summary(ret[-1], global_counter)
                    
                    if counter % summary_val_interval == 0:
                        print("summary_val_interval now")
                        batch = sample_test_data(val_dir, args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT)
                        
                        ret = model.validate_step(sess, batch, summary=True)
                        summary_writer.add_summary(ret[-1], global_counter)
                    
                    if check_if_should_pause(args.tag):
                        model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
                        print('pause and save model @ {} steps:{}'.format(save_model_dir, model.global_step.eval()))
                        sys.exit(0)
                            
                    batch_time = time.time()
                
                sess.run(model.epoch_add_op)
                
                model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
        
                # dump test data every 10 epochs
                if (epoch + 1) % 10 == 0:
                    os.makedirs(os.path.join(res_dir, str(epoch)), exist_ok=True)
                    os.makedirs(os.path.join(res_dir, str(epoch), 'data'), exist_ok=True)
                    
                    for batch in iterate_data(val_dir, shuffle=False, aug=False, is_testset=False, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):
                        
                        tags, results = model.predict_step(sess, batch, summary=False, vis=False)
                                
                        for tag, result in zip(tags, results):
                            of_path = os.path.join(res_dir, str(epoch), 'data', tag + '.txt')
                            with open(of_path, 'w+') as f:
                                labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                                for line in labels:
                                    f.write(line)
                                print('write out {} objects to {}'.format(len(labels), tag))



            # finally save model
            model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)


if __name__ == '__main__':
    tf.app.run(main)
