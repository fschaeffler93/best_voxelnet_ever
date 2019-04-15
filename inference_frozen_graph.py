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

def predict_frozen(graph, session, data, summary=False, vis=True):
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar = data[6]

        print('predict', tag)
        input_feed = {}
        input_feed[graph.get_tensor_by_name('prefix/phase:0')] = False
        for idx in range(1):
            input_feed[graph.get_tensor_by_name('prefix/gpu_0/feature:0')] = vox_feature[idx]
            input_feed[graph.get_tensor_by_name('prefix/gpu_0/coordinate:0')] = vox_coordinate[idx]

        output_feed = [graph.get_tensor_by_name('prefix/concat_101:0'), graph.get_tensor_by_name('prefix/concat_100:0')]
        probs, deltas = session.run(output_feed, input_feed)
        
        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(deltas, cal_anchors(), coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = probs.reshape((1 * 1, -1))

        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(1 * 1):
            # remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            ind = session.run(graph.get_tensor_by_name('prefix/non_max_suppression/NonMaxSuppressionV3:0'),
                    {graph.get_tensor_by_name('prefix/Placeholder_3:0'): boxes2d,
                     graph.get_tensor_by_name('prefix/Placeholder_4:0'): tmp_scores})

            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile('Car', len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))

        if vis:
            front_images, bird_views, heatmaps = [], [], []
            for i in range(len(img)):
                cur_tag = tag[i]
                P, Tr, R = load_calib( os.path.join( cfg.DATA_DIR, 'testing', 'calib', cur_tag + '.txt' ) )
                
                front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                                                 
                bird_view = lidar_to_bird_view_img(lidar[i], factor=cfg.BV_LOG_FACTOR)
                                                 
                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i], factor=cfg.BV_LOG_FACTOR, T_VELO_2_CAM=Tr, R_RECT_0=R)
                
                heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)
                
                front_images.append(front_image)
                bird_views.append(bird_view)
                heatmaps.append(heatmap)
            
            return tag, ret_box3d_score, front_images, bird_views, heatmaps

        return tag, ret_box3d_score

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

        calib_graph = load_graph(save_model_dir + "/frozen.pb")

        sess = tf.Session(config=conf, graph=calib_graph)

        for batch in iterate_data(test_dir, shuffle=False, aug=False, is_testset=True, batch_size=1, multi_gpu_sum=1):

            if args.vis:
                tags, results, front_images, bird_views, heatmaps = predict_frozen(calib_graph, sess, batch, summary=False, vis=True)
            else:
                tags, results = predict_frozen(sess, batch, summary=False, vis=False)
            
            for tag, result in zip(tags, results):
                of_path = os.path.join(args.output_path, 'data', tag + '.txt')
                with open(of_path, 'w+') as f:
                    labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                    for line in labels:
                        f.write(line)
                    print('write out {} objects to {}'.format(len(labels), tag))
            # dump visualizations
            if args.vis:
                for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                    front_img_path = os.path.join( args.output_path, 'vis', tag + '_front.jpg'  )
                    bird_view_path = os.path.join( args.output_path, 'vis', tag + '_bv.jpg'  )
                    heatmap_path = os.path.join( args.output_path, 'vis', tag + '_heatmap.jpg'  )
                    cv2.imwrite( front_img_path, front_image )
                    cv2.imwrite( bird_view_path, bird_view )
                    cv2.imwrite( heatmap_path, heatmap )


if __name__ == '__main__':
    tf.app.run(main)
