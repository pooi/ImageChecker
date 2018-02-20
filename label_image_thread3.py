# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, time
from PIL import Image


import numpy as np
import tensorflow as tf

from queue import Queue
from Checker2 import Checker


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    sess.close()

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def checkDir(dir_name, list):
    # print("Dir: ", dir_name)
    for dir in os.listdir(dir_name):
        if dir.startswith("."):
            continue

        path = os.path.join(dir_name, dir)
        if os.path.isdir(path):
            checkDir(path, list)
        else:
            fname, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext.endswith("jpeg") or ext.endswith("jpg") or ext.endswith("png") or ext.endswith("gif"):
                list.append(path)


def collectImage(dir_name, data):
    # print("Dir: ", dir_name)
    list = []
    for dir in os.listdir(dir_name):
        if dir.startswith("."):
            continue

        path = os.path.join(dir_name, dir)
        if os.path.isdir(path):
            collectImage(path, data)
        else:
            fname, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext.endswith("jpeg") or ext.endswith("jpg") or ext.endswith("png") or ext.endswith("gif"):
                list.append(path)

    dir_name = dir_name.replace("\\", "/")
    folder = dir_name.split("/").pop()
    if len(list) > 0 and (not folder.startswith(".")):
        data[folder] = list


if __name__ == "__main__":
    file_name = "images/test.jpg"
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Mul"
    output_layer = "final_result"

    dirname = "./"
    base_accuracy = 0.5
    num_of_thread = 4

    parser = argparse.ArgumentParser()
    # parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    # parser.add_argument("--input_height", type=int, help="input height")
    # parser.add_argument("--input_width", type=int, help="input width")
    # parser.add_argument("--input_mean", type=int, help="input mean")
    # parser.add_argument("--input_std", type=int, help="input std")
    # parser.add_argument("--input_layer", help="name of input layer")
    # parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--dir", help="name of directory")
    parser.add_argument("--accuracy", help="set accuracy criteria")
    parser.add_argument("--thread", help="number of thread")
    args = parser.parse_args()

    if args.dir:
        dirname = args.dir
    if args.accuracy:
        try:
            base_accuracy = float(args.accuracy)
        except:
            base_accuracy = 0.5
#        base_accuracy = args.accuracy
    if args.thread:
        try:
            num_of_thread = int(args.thread)
            if num_of_thread < 1:
                num_of_thread = 4
        except:
            num_of_thread = 4


    if args.graph:
        model_file = args.graph
    # if args.image:
    #   file_name = args.image
    if args.labels:
        label_file = args.labels
    # if args.input_height:
    #   input_height = args.input_height
    # if args.input_width:
    #   input_width = args.input_width
    # if args.input_mean:
    #   input_mean = args.input_mean
    # if args.input_std:
    #   input_std = args.input_std
    # if args.input_layer:
    #   input_layer = args.input_layer
    # if args.output_layer:
    #   output_layer = args.output_layer

    graph = load_graph(model_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    # t = read_tensor_from_image_file(file_name,
    #                                 input_height=input_height,
    #                                 input_width=input_width,
    #                                 input_mean=input_mean,
    #                                 input_std=input_std)
    labels = load_labels(label_file)

    # with tf.Session(graph=graph) as sess:
    try:
        data = {}
        collectImage(dirname, data)

        total_image_count = 0
        count = 0
        remove_count = 0
        remove_list = []
        for key in data.keys():
            total_image_count += len(data[key])

        if len(data.keys()) > 1:
            print("Found %d categories." % len(data.keys()))
        else:
            print("Found %d category." % len(data.keys()))
        if total_image_count > 1:
            print("Found %d images." % total_image_count)
        else:
            print("Found %d image." % total_image_count)

        queue = Queue()
        queueList = []
        remove_list = []

        for i in range(num_of_thread):
            checker = Checker(i, queue, graph, output_operation, input_operation, labels, base_accuracy, total_image_count, remove_list)
            checker.daemon = True
            checker.start()


        for key in data.keys():
            # print("Check", key)
            list = data[key]
            for fname in list:
                item = []
                item.append(key)
                item.append(fname)
                queueList.append(item)
                # queue.put(item)

        DIV = 100
        length = len(queueList)
        for step in range((length // DIV) + 1):
            print("Start", (step+1), "/", ((length // DIV) + 1))
            start_time = time.time()
            queue = Queue()
            threads = []
            sess = tf.Session(graph=graph)

            for i in range(num_of_thread):
                checker = Checker(i, queue, sess, output_operation, input_operation, labels, base_accuracy,
                                  DIV, remove_list)
                checker.daemon = True
                threads.append(checker)
                checker.start()

            for item in queueList[step * DIV : min((step + 1) * DIV, length)]:
                queue.put(item)


            queue.join()
            del threads
            sess.close()
            sess.__del__()
            print("Step", (step+1), "| Removed %d images" % len(remove_list))
            print("--- %s seconds ---" % (time.time() - start_time))
            print()

        print("Removed %d images" % len(remove_list))
    except Exception as e:
        print(e)
    # finally:
    #     sess.close()

        # results = sess.run(output_operation.outputs[0],
        #                   {input_operation.outputs[0]: t})
        # results = np.squeeze(results)
        #
        # top_k = results.argsort()[-5:][::-1]
        # labels = load_labels(label_file)
        # for i in top_k:
        #   print(labels[i], results[i])
