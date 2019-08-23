import cv2
import time
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.tensorrt
from core import utils
import json
import numpy as np
import datetime
import os
import base64
import requests

import multiprocessing as mp

from flask import Flask, request, jsonify
flaskserver = Flask(__name__)

from args import imgsize, batchsize

global detection

# Handle API Call
@flaskserver.route(rule='/detect', methods=['POST'])
def detect():
    
    global detection
    content = request
    batch = np.fromstring(content.data, np.float32).reshape((-1, imgsize, imgsize, 3))
    inferenceList = detection.inference(batch)
    return jsonify({"Response":str(inferenceList)})

class Detection():
    ''' Class to hold all detection related member methods and variables. '''

    def __init__(self):
        ''' Called when class object is created. '''

        self.img_size = imgsize
        self.max_batch_size = batchsize
        self.num_classes = len(utils.read_coco_names('./data/coco.names'))

        self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./checkpoint/yolov3_cpu_nms.pb",
                                                ["Placeholder:0", "concat_9:0", "mul_6:0"])

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=self.config)
        _ = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: self.createRandomSample()})

    def createRandomSample(self):
        '''Method to create random sample to initialize tensorflow session. '''

        randomSample = None
        for i in range(0, self.max_batch_size):
            if i == 0:
                randomSample = np.expand_dims(np.random.random_sample((self.img_size, self.img_size, 3)), axis=0)
                continue
            randomSample = np.concatenate((randomSample, np.expand_dims(np.random.random_sample((self.img_size, self.img_size, 3)), axis=0)), axis=0)
        return randomSample
        
    def inference(self, batch):
        ''' Method to perform inference on a batch. '''

        inferenceList = list()
        start = time.time()
        prev_time = time.time()
        boxes, scores = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: batch})
        for i in range(0, len(scores)):
            t_boxes = boxes[i]
            t_boxes = np.expand_dims(t_boxes, axis=0)
            t_scores = scores[i]
            t_scores = np.expand_dims(t_scores, axis=0)
            t_boxes, t_scores, t_labels = utils.cpu_nms(t_boxes, t_scores, self.num_classes, score_thresh=0.5, iou_thresh=0.4)
            inferenceList.append([t_labels, t_scores, t_boxes])
        print(time.time() - start)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        info = "time: %.2f ms" % (1000 * exec_time)
        
        return inferenceList

detection = Detection()

if __name__ == '__main__':

    flaskserver.run(host='127.0.0.1',
                    port=5000,
                    debug=True)