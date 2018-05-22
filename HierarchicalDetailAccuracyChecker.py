import os, gc, shutil
from PIL import Image

import numpy as np
import tensorflow as tf

from threading import Thread
from queue import Queue


class Checker(Thread):

    def __init__(self, num, queue, sess, output_operation, input_operation, labels, detail_graphs, check_rates, hierarchical, base_accuracy, total_count, remove_list, summary):
        Thread.__init__(self)
        self.num = num
        self.queue = queue
        self.sess = sess
        self.output_operation = output_operation
        self.input_operation = input_operation
        self.labels = labels
        self.detail_graphs = detail_graphs
        self.base_accuracy = base_accuracy
        self.total_count = total_count
        self.remove_list = remove_list
        self.summary = summary
        self.count = 0
        self.check_rates = check_rates
        self.hierarchical = hierarchical


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
                image = Image.open(fname)
                image.load()
                image.close()
                format = image.format
                if format == "JPEG" or format == "PNG":
                    pass
                else:
                    print(self.num, "Not collect:", fname, " | Format error")
                    return
            except:
                print(self.num, "Not collect:", fname, " | Format error")
                return

            t = self.read_tensor_from_image_file(fname)

            results = self.sess.run(self.output_operation.outputs[0],
                               {self.input_operation.outputs[0]: t})

            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            # for i in top_k:
            #     print(labels[i], results[i])

            # print(self.num, "9")
            if len(top_k) < 0:
                print(self.num, "Not collect:", fname, " | Can't recognition")

            else:
                # isFinish = False
                top_i = top_k[0]
                top_label = self.labels[top_i]
                # top_result = results[top_i]

                second_i = top_k[1]
                second_label = self.labels[second_i]

                if key in self.hierarchical[top_label]: # 1차 분류 성공

                    hierarchical_label = top_label
                    hierarchical_index = int(hierarchical_label.replace("classification", ""))

                    data = self.detail_graphs[hierarchical_index-1]
                    detail_results = data['sess'].run(data['output_operation'].outputs[0],
                                            {data['input_operation'].outputs[0]: t})
                    detail_results = np.squeeze(detail_results)
                    detail_top_k = detail_results.argsort()[-5:][::-1]

                    if len(detail_top_k) < 0:
                        print(self.num, "Not collect:", fname, " | Can't recognition")
                    else:
                        detail_top_i = detail_top_k[0]
                        detail_top_label = data['labels'][detail_top_i]
                        detail_top_result = detail_results[detail_top_i]

                        if detail_top_label == key:
                            for s in self.check_rates:
                                collect = 'collect_' + str(s)
                                notcollect = 'notcollect_' + str(s)
                                if detail_top_result >= s/100:
                                    self.summary[key][collect].append(self.num)
                                else:
                                    self.summary[key][notcollect].append(self.num)
                        else:
                            for s in self.check_rates:
                                notcollect = 'notcollect_' + str(s)
                                self.summary[key][notcollect].append(self.num)

                else:
                    for s in self.check_rates:
                        notcollect = 'notcollect_' + str(s)
                        self.summary[key][notcollect].append(self.num)
                # else: # 1차 분류 실패
                #     # for s in [20, 30, 40, 50, 60, 70, 80, 90, 95]:
                #     #     notcollect = 'notcollect_' + str(s)
                #     #     self.summary[key][notcollect].append(self.num)
                #
                #     if key in self.hierarchical[second_label]:
                #         self.summary[key]["classification_second"].append(self.num)
                #
                #         # ====================================
                #
                #         hierarchical_label = top_label
                #         hierarchical_index = int(hierarchical_label.replace("classification", ""))
                #
                #         data = self.detail_graphs[hierarchical_index - 1]
                #         detail_results_1 = data['sess'].run(data['output_operation'].outputs[0],
                #                                           {data['input_operation'].outputs[0]: t})
                #         detail_results_1 = np.squeeze(detail_results_1)
                #         detail_top_k_1 = detail_results_1.argsort()[-5:][::-1]
                #
                #         detail_top_i_1 = detail_top_k_1[0]
                #         detail_top_label_1 = data['labels'][detail_top_i_1]
                #         detail_top_result_1 = detail_results_1[detail_top_i_1]
                #
                #
                #         # ====================================
                #
                #         hierarchical_label = second_label
                #         hierarchical_index = int(hierarchical_label.replace("classification", ""))
                #
                #         data = self.detail_graphs[hierarchical_index - 1]
                #         detail_results_2 = data['sess'].run(data['output_operation'].outputs[0],
                #                                           {data['input_operation'].outputs[0]: t})
                #         detail_results_2 = np.squeeze(detail_results_2)
                #         detail_top_k_2 = detail_results_2.argsort()[-5:][::-1]
                #
                #         detail_top_i_2 = detail_top_k_2[0]
                #         detail_top_label_2 = data['labels'][detail_top_i_2]
                #         detail_top_result_2 = detail_results_2[detail_top_i_2]
                #
                #         # ====================================
                #         compare1 = float(detail_top_result_1)
                #         compare2 = float(detail_top_result_2)
                #         # print(float(detail_top_result_1))
                #         # print(detail_top_result_2)
                #         # print(detail_top_result_1 >= detail_top_result_2)
                #
                #         if compare1 >= compare2:
                #             detail_top_label = detail_top_label_1
                #             detail_top_result = compare1
                #         else:
                #             detail_top_label = detail_top_label_2
                #             detail_top_result = compare2
                #
                #         # print(str(detail_top_label))
                #         # print(detail_top_label)
                #         # print(detail_top_label == key)
                #
                #         if detail_top_label == key:
                #             for s in [20, 30, 40, 50, 60, 70, 80, 90, 95]:
                #                 collect = 'collect_' + str(s)
                #                 notcollect = 'notcollect_' + str(s)
                #                 if detail_top_result >= s / 100:
                #                     self.summary[key][collect].append(self.num)
                #                 else:
                #                     self.summary[key][notcollect].append(self.num)
                #         else:
                #             for s in [20, 30, 40, 50, 60, 70, 80, 90, 95]:
                #                 notcollect = 'notcollect_' + str(s)
                #                 self.summary[key][notcollect].append(self.num)
                #
                #
                #     else:
                #         for s in [20, 30, 40, 50, 60, 70, 80, 90, 95]:
                #             notcollect = 'notcollect_' + str(s)
                #             self.summary[key][notcollect].append(self.num)
                #         print(fname)

                self.summary[key][top_label].append(self.num)

                del results
                results = None

            del t
            t = None
        except Exception as e:
            print(e)
            try:
                print(self.num, "Not collect:", fname, " | Unknown error")
            except:
                pass


    def run(self):
        while True:
            self.count += 1
            if self.count % 13 == 0:
                pass

            data = self.queue.get()
            if (self.total_count - self.queue.qsize()) % 5 == 0:
                print(self.num, "Completed %d/%d" % (self.total_count - self.queue.qsize(), self.total_count))

            self.check_image(data[0], data[1])
            self.queue.task_done()
