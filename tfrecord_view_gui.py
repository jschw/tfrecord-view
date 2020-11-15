import sys
import os
import argparse

from PyQt5.QtWidgets import (
    QWidget, QPushButton,
    QLabel, QApplication,
    QFileDialog)
from PyQt5.QtGui import (QFont, QPixmap, QImage)

import cv2
import numpy as np
import tensorflow as tf


class TfrecordBrowser(QWidget):

    img_width = 0
    img_height = 0
    img_scale = 1.0

    tfrecord_path = ''
    pbtxt_path = ''

    img_array = []
    actual_sample_no = 0

    btn_offset = 40
    no_labelmap = True


    def browseFile(self, title):

        dirname, filename = os.path.split(os.path.abspath(__file__))

        fname = QFileDialog.getOpenFileName(self, title, dirname)

        return fname[0]


    def cv_bbox(self, image, bbox, color = (255, 255, 255), line_width = 2):

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_width)
        return


    def parse_record(self, data_record):

        feature = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                   'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                   'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                   'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                   'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                   'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                   'image/filename': tf.io.FixedLenFeature([], tf.string)
                   }
        return tf.io.parse_single_example(data_record, feature)


    def load_records(self, file_path, class_labels, stride = 1):

        dataset = tf.data.TFRecordDataset([file_path])
        record_iterator = iter(dataset)
        num_records = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()


        for im_ind in range(num_records):

            parsed_example = self.parse_record(record_iterator.get_next())
            if im_ind % stride != 0:
                continue

            fname = parsed_example['image/filename'].numpy()
            encoded_image = parsed_example['image/encoded']
            image_np = tf.image.decode_image(encoded_image, channels=3).numpy()

            labels =  tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=0).numpy()
            x1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmin'], default_value=0).numpy()
            x2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmax'], default_value=0).numpy()
            y1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymin'], default_value=0).numpy()
            y2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymax'], default_value=0).numpy()

            num_bboxes = len(labels)

            height, width = image_np[:, :, 1].shape
            self.img_height = height
            self.img_width = width
            image_copy = image_np.copy()
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            if num_bboxes > 0:
                x1 = np.int64(x1norm*width)
                x2 = np.int64(x2norm*width)
                y1 = np.int64(y1norm*height)
                y2 = np.int64(y2norm*height)
                for bbox_ind in range(num_bboxes):
                        bbox = (x1[bbox_ind], y1[bbox_ind], x2[bbox_ind], y2[bbox_ind])
                        if not self.no_labelmap:
                            label_name = list(class_labels.keys())[list(class_labels.values()).index(labels[bbox_ind])]
                            label_position = (bbox[0] + 5, bbox[1] + 20)
                            cv2.putText(image_rgb,
                                    label_name,
                                    label_position,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 10, 10), 2); #scale, color, thickness

                        self.cv_bbox(image_rgb, bbox, color = (0, 250, 0), line_width = 2)
                        


            if self.img_scale != 1.0:
                image_rgb = cv2.resize(image_rgb, (int(self.img_width*self.img_scale), int(self.img_height*self.img_scale)), interpolation = cv2.INTER_AREA)

            self.img_array.append(image_rgb)
            
        
        print('\n==> All data read in.')
        
        print('==> Total count of samples in dataset: ' + str(len(self.img_array)-1) + '\n')



    def __init__(self, scale_factor):
        super(TfrecordBrowser, self).__init__()

        self.img_scale = scale_factor

        self.tfrecord_path = self.browseFile('Open .tfrecord file')

        if not self.tfrecord_path:
            print('==> No .tfrecord file selected.')
            quit()

        print('\n==> Selected .tfrecord file: ' + self.tfrecord_path)

        self.pbtxt_path =  self.browseFile('Open .pbtxt class label file')

        if not self.pbtxt_path:
            print('==> No .pbtxt file selected.')
            self.no_labelmap = True
        else: self.no_labelmap = False

        print('==> Selected .pbtxt file: ' + self.pbtxt_path + '\n')


        # Read class label pbtxt file
        class_labels = {}
        if not self.no_labelmap:
            pbtxt_file = open(self.pbtxt_path, 'r') 
            lines = pbtxt_file.readlines() 
              
            count = 1
            for line in lines: 
                if 'name' in line:
                    classname = line.split('\'')[1]
                    class_labels[classname] = count
                    count = count+1


        self.load_records(self.tfrecord_path, class_labels, stride = 1)

        self.initUI()


    def initUI(self):

        img_width = self.img_array[0].shape[1]
        img_height = self.img_array[0].shape[0]

        # Show  image
        self.pic = QLabel(self)
        # (spacing x, spacing y, image x, image y)
        self.pic.setGeometry(self.btn_offset, 0, img_width, img_height)

        img_array = cv2.cvtColor(self.img_array[self.actual_sample_no], cv2.COLOR_BGR2RGB)

        img_qt = QImage(img_array, img_array.shape[1], img_array.shape[0], 3*img_array.shape[1], QImage.Format_RGB888)
        img_pxmap = QPixmap.fromImage(img_qt)
        self.pic.setPixmap(img_pxmap)

        # Previous button 
        self.btn_prev = QPushButton('<<', self)
        self.btn_prev.setToolTip('Go to the <b>Previous</b> image in the dataset.')
        self.btn_prev.resize(self.btn_prev.sizeHint())
        self.btn_prev.setGeometry(0, 0, self.btn_offset, img_height)
        self.btn_prev.clicked.connect(self.prev_entry)

        # Next button 
        self.btn_next = QPushButton('>>', self)
        self.btn_next.setToolTip('Go to <b>Next</b> image in the dataset.')
        self.btn_next.resize(self.btn_next.sizeHint())
        self.btn_next.setGeometry(self.btn_offset+img_width, 0, self.btn_offset, img_height)
        self.btn_next.clicked.connect(self.next_entry)

        self.setGeometry(300, 300, img_width+2*self.btn_offset, img_height)
        self.setWindowTitle('Browse file: ' + self.tfrecord_path)
        self.show()

    # Connect buttons to image updating 
    def next_entry(self):
        if self.actual_sample_no+1 == len(self.img_array)-1:
            return

        self.actual_sample_no = self.actual_sample_no + 1

        img_width = self.img_array[self.actual_sample_no].shape[1]
        img_height = self.img_array[self.actual_sample_no].shape[0]

        self.pic.setGeometry(self.btn_offset, 0, img_width, img_height)

        self.btn_prev.setGeometry(0, 0, self.btn_offset, img_height)
        self.btn_next.setGeometry(self.btn_offset+img_width, 0, self.btn_offset, img_height)

        img_array = cv2.cvtColor(self.img_array[self.actual_sample_no], cv2.COLOR_BGR2RGB)

        img_qt = QImage(img_array, img_array.shape[1], img_array.shape[0], 3*img_array.shape[1], QImage.Format_RGB888)
        img_pxmap = QPixmap.fromImage(img_qt)

        self.pic.setPixmap(QPixmap(img_pxmap))

        self.resize(2*self.btn_offset+img_width, img_height)



    def prev_entry(self):
        if self.actual_sample_no-1 <0:
            return

        self.actual_sample_no = self.actual_sample_no - 1

        img_width = self.img_array[self.actual_sample_no].shape[1]
        img_height = self.img_array[self.actual_sample_no].shape[0]

        self.pic.setGeometry(self.btn_offset, 0, img_width, img_height)

        self.btn_prev.setGeometry(0, 0, self.btn_offset, img_height)
        self.btn_next.setGeometry(self.btn_offset+img_width, 0, self.btn_offset, img_height)

        img_array = cv2.cvtColor(self.img_array[self.actual_sample_no], cv2.COLOR_BGR2RGB)

        img_qt = QImage(img_array, img_array.shape[1], img_array.shape[0], 3*img_array.shape[1], QImage.Format_RGB888)
        img_pxmap = QPixmap.fromImage(img_qt)

        self.pic.setPixmap(QPixmap(img_pxmap))

        self.resize(2*self.btn_offset+img_width, img_height)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scale down images with this factor.')
    args = parser.parse_args()

    scale_factor = args.scale

    app = QApplication(sys.argv)
    ex = TfrecordBrowser(scale_factor)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()