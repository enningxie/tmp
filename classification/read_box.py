# azt
import os
import cv2
import pickle
import tensorflow as tf


def input_data():
    # filename = '/var/Data/xz/butterfly/det_test_btf.txt'
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
            name_1, _, xmin_1, ymin_1, xmax_1, ymax_1 = [i for i in lines.split()]
            name.append(name_1)
            xmin.append(float(xmin_1))
            ymin.append(float(ymin_1))
            xmax.append(float(xmax_1))
            ymax.append(float(ymax_1))
    return name, xmin, ymin, xmax, ymax


def box_image(name, xmin, ymin, xmax, ymax):
    image_dir = '/home/enningxie/Documents/DataSets/butter_data/JPEGImages'
    # saved_image_data = '../butter_faster_rcnnimg.dat'
    # saved_file_names = '../test.txt'
    # to_path = '../savedImages/'
    name_2 = ''
    image_data = []
    file_names = []
    # Load face encodings
    # with open(saved_image_data, 'rb') as f:
    #     images_data = pickle.load(f)
    # with open(saved_file_names, 'r') as f:
    #     name_list = f.readlines()
    for name_1, xmin_1, ymin_1, xmax_1, ymax_1 in zip(name, xmin, ymin, xmax, ymax):
        if name_1 != name_2:
            name_2 = name_1
            # image = images_data[name_list.index(name_1.split('.')[0]+'\n')]
            image_path = os.path.join(image_dir, name_1 + '.jpg')
            file_names.append(name_1)
            image = cv2.imread(image_path)
            image_cp = image[int(ymin_1): int(ymax_1), int(xmin_1): int(xmax_1)]
            img_str = cv2.imencode('.jpg', image_cp)[1].tostring()
            img_tensor = tf.convert_to_tensor(img_str, dtype=tf.string)
            # cv2.imwrite(os.path.join(to_path, name_1), image_cp)
            image_data.append(img_tensor)
    return file_names, image_data


def get_tiny_image():
    n, x1, y1, x2, y2 = input_data()
    return box_image(n, x1, y1, x2, y2)

if __name__ == '__main__':
    saved_file_names = '../test.txt'
    name_list = []
    with open(saved_file_names, 'r') as f:
        while f.readline() != '':
            name_list.append(f.readline().strip())
    print(name_list)

