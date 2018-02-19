import os
from PIL import Image

import numpy as np
import tensorflow as tf

from threading import Thread
from queue import Queue


class Checker(Thread):

    def __init__(self, queue, sess, output_operation, input_operation, labels, base_accuracy, total_count, remove_list):
        Thread.__init__(self)
        self.queue = queue
        self.sess = sess
        self.output_operation = output_operation
        self.input_operation = input_operation
        self.labels = labels
        self.base_accuracy = base_accuracy
        self.total_count = total_count
        self.remove_list = remove_list

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
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def check_image(self, key, fname):

        try:
            try:
                image = Image.open(fname)
                image.load()
                image.close()
                format = image.format
                if format == "JPEG" or format == "PNG":
                    pass
                else:
                    print("Del:", fname, " | Format error")
                    os.remove(fname)
                    self.remove_list.append(fname)
                    return
            except:
                print("Del:", fname, " | Format error")
                os.remove(fname)
                self.remove_list.append(fname)
                return

            t = self.read_tensor_from_image_file(fname)

            results = self.sess.run(self.output_operation.outputs[0],
                               {self.input_operation.outputs[0]: t})
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            # for i in top_k:
            #     print(labels[i], results[i])

            if len(top_k) < 0:
                print("Del:", fname, " | Can't recognition")
                os.remove(fname)
                self.remove_list.append(fname)
            else:
                isFinish = False
                top_i = top_k[0]
                for i in top_k:
                    label = self.labels[i]
                    result = results[i]

                    if label == key:
                        if result < self.base_accuracy:
                            print("Del:", fname, " | ", result)
                            os.remove(fname)
                            self.remove_list.append(fname)
                            isFinish = True
                        else:
                            isFinish = True

                    if isFinish:
                        break
                if not isFinish:
                    print("Del:", fname, " | Recognition fail -", self.labels[top_i])
                    os.remove(fname)
                    self.remove_list.append(fname)
        except Exception as e:
            print(e)
            try:
                os.remove(fname)
                print("Del:", fname, " | Unknown error")
                self.remove_list.append(fname)
            except:
                pass


    def run(self):
        while True:
            data = self.queue.get()
            if (self.total_count - self.queue.qsize()) % 5 == 0:
                print("Completed %d/%d" % (self.total_count - self.queue.qsize(), self.total_count), "| ", "Removed %d images" % len(self.remove_list))
            self.check_image(data[0], data[1])
            self.queue.task_done()