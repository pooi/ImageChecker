import os, gc
from PIL import Image

import numpy as np
import tensorflow as tf

from threading import Thread
from queue import Queue


class Checker(Thread):

    def __init__(self, num, queue, graph, output_operation, input_operation, labels, base_accuracy, total_count, remove_list):
        Thread.__init__(self)
        self.num = num
        self.queue = queue
        self.graph = graph
        self.sess = tf.Session(graph=self.graph)
        self.output_operation = output_operation
        self.input_operation = input_operation
        self.labels = labels
        self.base_accuracy = base_accuracy
        self.total_count = total_count
        self.remove_list = remove_list
        self.count = 0

    def read_tensor_from_image_file(self, file_name, input_height=299, input_width=299,
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
        del file_reader
        file_reader = None
        float_caster = tf.cast(image_reader, tf.float32)
        del image_reader
        image_reader = None
        dims_expander = tf.expand_dims(float_caster, 0)
        del float_caster
        float_caster = None
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        del dims_expander
        dims_expander = None
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        del resized
        resized = None
        with tf.Session() as sess:
            result = sess.run(normalized)
            del normalized
            normalized = None
            sess.close()
            sess = None

        return result

    def getMoveDir(self, fname):
        fname = str(fname)
        tempDirL = fname.split('/')[:-2]
        tempDirR = fname.split('/')[-2:]

        new_dir = ""
        for i in range(len(tempDirL)):
            new_dir += tempDirL[i]
            if i + 1 < len(tempDirL):
                new_dir += '/'
            else:
                new_dir += '_selected'

            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
                print("Create dir:", new_dir)

        file_dir = ""
        for i in range(len(tempDirR)):
            file_dir += tempDirR[i]
            if i + 1 < len(tempDirR):
                file_dir += '/'

        move_dir = os.path.join(new_dir, file_dir)
        dirname = os.path.dirname(move_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print("Create dir:", dirname)

        return move_dir

    def check_image(self, key, fname):

        try:
            try:
                # print(self.num, "1")
                image = Image.open(fname)
                image.load()
                image.close()
                format = image.format
                if format == "JPEG" or format == "PNG":
                    # print(self.num, "2")
                    pass
                else:
                    print(self.num, "Del:", fname, " | Format error")
                    os.remove(fname)
                    self.remove_list.append(self.num)
                    # print(self.num, "3")
                    return
            except:
                print(self.num, "Del:", fname, " | Format error")
                os.remove(fname)
                self.remove_list.append(self.num)
                # print(self.num, "4")
                return

            # print(self.num, "5")
            t = self.read_tensor_from_image_file(fname)
            # print(self.num, "6")

            results = self.sess.run(self.output_operation.outputs[0],
                               {self.input_operation.outputs[0]: t})
            del t
            t = None
            # print(self.num, "7")
            results = np.squeeze(results)
            # print(self.num, "8")

            top_k = results.argsort()[-5:][::-1]
            # for i in top_k:
            #     print(labels[i], results[i])

            # print(self.num, "9")
            if len(top_k) < 0:
                print(self.num, "Del:", fname, " | Can't recognition")
                os.remove(fname)
                self.remove_list.append(self.num)
                # print(self.num, "10")
            else:
                isFinish = False
                top_i = top_k[0]
                for i in top_k:
                    label = self.labels[i]
                    result = results[i]

                    if label == key:
                        if result < self.base_accuracy:
                            print(self.num, "Del:", fname, " | ", result)
                            os.remove(fname)
                            self.remove_list.append(self.num)
                            isFinish = True
                        else:
                            new_dir = self.getMoveDir(fname)
                            os.rename(fname, new_dir)
                            print(self.num, "Move:", fname, "to", new_dir)
                            isFinish = True

                    if isFinish:
                        break
                # print(self.num, "11")
                del results
                results = None
                if not isFinish:
                    print(self.num, "Del:", fname, " | Recognition fail -", self.labels[top_i])
                    os.remove(fname)
                    self.remove_list.append(self.num)
                # print(self.num, "12")
        except Exception as e:
            print(e)
            try:
                os.remove(fname)
                print(self.num, "Del:", fname, " | Unknown error")
                self.remove_list.append(self.num)
            except:
                pass


    def run(self):
        while True:
            self.count += 1
            if self.count % 13 == 0:
                self.sess.close()
                del self.sess
                self.sess = None
                self.sess = tf.Session(graph=self.graph)
                print(self.num, "Session create.", self.sess)
                gc.collect()
                print(self.num, "Execute Garbage Collector")
            # if self.count % 100 == 0:
            #     gc.collect()
            #     print(self.num, "Execute Garbage Collector")
            data = self.queue.get()
            if (self.total_count - self.queue.qsize()) % 5 == 0:
                print(self.num, "Completed %d/%d" % (self.total_count - self.queue.qsize(), self.total_count), "| ", "Removed %d images" % len(self.remove_list))

            self.check_image(data[0], data[1])
            self.queue.task_done()