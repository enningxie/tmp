# azt
import os
import cv2
import tensorflow as tf


def input_data():
    filename = '/home/enningxie/Documents/DataSets/butter_data/det_test_btf.txt'
    name = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            name_1, xmin_1, ymin_1, xmax_1, ymax_1 = [i for i in lines.split()]
            name.append(name_1)
            xmin.append(float(xmin_1))
            ymin.append(float(ymin_1))
            xmax.append(float(xmax_1))
            ymax.append(float(ymax_1))
    return name, xmin, ymin, xmax, ymax


def box_image(name, xmin, ymin, xmax, ymax):
    image_dir = '/home/enningxie/Documents/DataSets/butter_data/JPEGImages'
    name_2 = ''
    image_data = []
    file_names = []
    for name_1, xmin_1, ymin_1, xmax_1, ymax_1 in zip(name, xmin, ymin, xmax, ymax):
        if name_1 != name_2:
            name_2 = name_1
            image_path = os.path.join(image_dir, name_1)
            file_names.append(name_1)
            image = cv2.imread(image_path)
            image_cp = image[int(ymin_1): int(ymax_1), int(xmin_1): int(xmax_1)]
            img_str = cv2.imencode('.jpg', image_cp)[1].tostring()
            img_tensor = tf.convert_to_tensor(img_str, dtype=tf.string)
            image_data.append(img_tensor)
    return file_names, image_data


def get_tiny_image():
    n, x1, y1, x2, y2 = input_data()
    return box_image(n, x1, y1, x2, y2)