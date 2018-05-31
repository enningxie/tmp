# xz
import argparse
import sys
import os
import tensorflow as tf
from read_box import get_tiny_image


FLAGS = None


def arger():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', default='./saveImages', type=str, help='Absolute path to image file.')
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=1,
        help='Display this many predictions.')
    parser.add_argument(
        '--graph',
        default='../model/output_graph.pb',
        type=str,
        help='Absolute path to graph file (.pb)')
    parser.add_argument(
        '--labels',
        default='../model/output_labels.txt',
        type=str,
        help='Absolute path to labels file (.txt)')
    parser.add_argument(
        '--output_layer',
        type=str,
        default='final_result:0',
        help='Name of the result operation')
    parser.add_argument(
        '--input_layer',
        type=str,
        default='DecodeJpeg/contents:0',
        help='Name of the input operation')
    return parser.parse_known_args()


def load_image(filename):
    return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label


def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        preds = []
        logits = []
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        for image in image_data:
            predictions, = sess.run(softmax_tensor, {input_layer_name: image})
            preds.append(predictions)

        # Sort to show labels in order of confidence
        for predictions in preds:
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            logits.append(top_k[0])

        return logits


def main(_):
    if not tf.gfile.Exists(FLAGS.image):
        tf.logging.fatal('image file does not exist %s', FLAGS.image)

    if not tf.gfile.Exists(FLAGS.labels):
        tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

    if not tf.gfile.Exists(FLAGS.graph):
        tf.logging.fatal('graph file does not exist %s', FLAGS.graph)


    # load labels
    labels = load_labels(FLAGS.labels)
    # load image
    file_names, image_data = get_tiny_image()

    # load graph, which is stored in the default session
    load_graph(FLAGS.graph)

    logits = run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
              FLAGS.num_top_predictions)

    with open('../A196_task2.txt', 'a') as f:
        for file_name, label_index in zip(file_names, logits):
            str_ = file_name + ' ' + labels[label_index] + '\n'
            f.write(str_)


if __name__ == '__main__':
    FLAGS, unparsed = arger()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
