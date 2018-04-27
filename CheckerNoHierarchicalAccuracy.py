import os, gc, shutil
from PIL import Image

import numpy as np
import tensorflow as tf

from threading import Thread
from queue import Queue


class Checker(Thread):

    def __init__(self, num, queue, sess, output_operation, input_operation, labels, base_accuracy, total_count, remove_list, summary):
        Thread.__init__(self)
        self.num = num
        self.queue = queue
        self.sess = sess
        self.output_operation = output_operation
        self.input_operation = input_operation
        self.labels = labels
        self.base_accuracy = base_accuracy
        self.total_count = total_count
        self.remove_list = remove_list
        self.summary = summary
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
        sess = tf.Session()
        # with tf.Session() as sess:
        result = sess.run(normalized)
        del normalized
        normalized = None
        sess.close()
        del sess
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
                    print(self.num, "Not collect:", fname, " | Format error")
                    # os.remove(fname)
                    # self.remove_list.append(self.num)
                    # self.summary[key]['notcollect'].append(self.num)
                    # print(self.num, "3")
                    return
            except:
                print(self.num, "Not collect:", fname, " | Format error")
                # os.remove(fname)
                # self.remove_list.append(self.num)
                # self.summary[key]['notcollect'].append(self.num)
                # print(self.num, "Del:", fname, " | Format error")
                # os.remove(fname)
                # self.remove_list.append(self.num)
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
                print(self.num, "Not collect:", fname, " | Can't recognition")
                # os.remove(fname)
                # self.remove_list.append(self.num)
                # self.summary[key]['notcollect'].append(self.num)
                # print(self.num, "Del:", fname, " | Can't recognition")
                # os.remove(fname)
                # self.remove_list.append(self.num)
                # print(self.num, "10")
            else:
                # isFinish = False
                top_i = top_k[0]

                top_label = self.labels[top_i]
                top_result = results[top_i]

                if key == top_label:
                    for s in [20, 30, 40, 50, 60, 70, 80, 90, 95]:
                        collect = 'collect_' + str(s)
                        notcollect = 'notcollect_' + str(s)
                        if top_result >= s/100:
                            self.summary[key][collect].append(self.num)
                        else:
                            self.summary[key][notcollect].append(self.num)
                else:
                    for s in [20, 30, 40, 50, 60, 70, 80, 90, 95]:
                        notcollect = 'notcollect_' + str(s)
                        self.summary[key][notcollect].append(self.num)

                # self.summary[key][top_label].append(self.num)

                # for i in top_k:
                #     label = self.labels[i]
                #     result = results[i]
                #
                #     if label == key:
                #
                #         if result < 0.2:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_20'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_20'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.3:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_30'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_30'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.4:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_40'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_40'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.5:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_50'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_50'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.6:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_60'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_60'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.7:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_70'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_70'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.8:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_80'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_80'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.9:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_90'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_90'].append(self.num)
                #             isFinish = True
                #
                #         if result < 0.95:
                #             # print(self.num, "Not collect:", fname, " | ", result)
                #             self.summary[key]['notcollect_95'].append(self.num)
                #             isFinish = True
                #         else:
                #             # print(self.num, "Collect:", fname, " | ", result)
                #             self.summary[key]['collect_95'].append(self.num)
                #             isFinish = True
                #
                #     if isFinish:
                #         break
                # print(self.num, "11")
                del results
                results = None
                # if not isFinish:
                #     print(self.num, "Not collect:", fname, " | Recognition fail -", self.labels[top_i])
                    # os.remove(fname)
                    # self.remove_list.append(self.num)
                    # self.summary[key]['notcollect'].append(self.num)
                    # print(self.num, "Del:", fname, " | Recognition fail -", self.labels[top_i])
                    # os.remove(fname)
                    # self.remove_list.append(self.num)
                # print(self.num, "12")
        except Exception as e:
            print(e)
            try:
                print(self.num, "Not collect:", fname, " | Unknown error")
                # os.remove(fname)
                # self.remove_list.append(self.num)
                # self.summary[key]['notcollect'].append(self.num)
                # os.remove(fname)
                # print(self.num, "Del:", fname, " | Unknown error")
                # self.remove_list.append(self.num)
            except:
                pass


    def run(self):
        while True:
            self.count += 1
            if self.count % 13 == 0:
                # self.sess.close()
                # del self.sess
                # self.sess = None
                # self.sess = tf.Session(graph=self.graph)
                # print(self.num, "Session create.", self.sess)
                # gc.collect()
                # print(self.num, "Execute Garbage Collector")
                pass
            # if self.count % 100 == 0:
            #     gc.collect()
            #     print(self.num, "Execute Garbage Collector")
            data = self.queue.get()
            if (self.total_count - self.queue.qsize()) % 5 == 0:
                print(self.num, "Completed %d/%d" % (self.total_count - self.queue.qsize(), self.total_count))


            self.check_image(data[0], data[1])
            self.queue.task_done()