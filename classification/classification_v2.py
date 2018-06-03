import argparse
import sys
import os
import tensorflow as tf
from read_box import get_tiny_image

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image', default='../savedImages/', type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=1,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    default='../new_model/output_graph.pb',
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    default='../new_model/output_labels.txt',
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
    default='Placeholder:0',
    help='Name of the input operation')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

def read_tensor_from_image_file(file_name,
                                input_height=331,
                                input_width=331,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  # file_reader = tf.read_file(file_name, input_name)
  # if file_name.endswith(".png"):
  #   image_reader = tf.image.decode_png(
  #       file_name, channels=3, name="png_reader")
  # elif file_name.endswith(".gif"):
  #   image_reader = tf.squeeze(
  #       tf.image.decode_gif(file_name, name="gif_reader"))
  # elif file_name.endswith(".bmp"):
  #   image_reader = tf.image.decode_bmp(file_name, name="bmp_reader")
  # else:
  image_reader = tf.image.decode_jpeg(
      file_name, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  return result


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    fix_graph_def(graph_def)
    tf.import_graph_def(graph_def, name='')


def run_graph(graph_cur, image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session(graph=graph_cur) as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def fix_graph_def(graph_def):
    # fix nodes
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            print('+++++')
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            print('++')
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                print('++++++++++++++++++++++++++++++++++++')
                del node.attr['use_locking']
        if "dilations" in node.attr:
            print('dasdasdasdasd')
            del node.attr["dilations"]
        if "index_type" in node.attr:
            print('++++++++++++++++++++++++++++++++++++++++++++')
            del node.attr["index_type"]


def main(_):
  """Runs inference on an image."""

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  # load image
  # image_data = load_image(FLAGS.image)
  # load image
  file_names, image_data = get_tiny_image()
  # get_tiny_image()
  images_data = []
  # image_files = os.listdir(FLAGS.image)
  for image_file in image_data:
      # image_path = os.path.join(FLAGS.image, image_file)
      t = read_tensor_from_image_file(
          image_file,
          input_height=224,
          input_width=224,
          input_mean=0,
          input_std=255)
      images_data.append(t)



  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)


  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.output_layer)
    preds = []
    for image_data_ in images_data:
        predictions, = sess.run(softmax_tensor, {FLAGS.input_layer: image_data_})
        preds.append(predictions)

    # Sort to show labels in order of confidence
    logits = []
    for pred in preds:
        top_k = pred.argsort()[-FLAGS.num_top_predictions:][::-1]
        logits.append(top_k[0])

    with open('../A196_task2_.txt', 'a') as f:
        for file_name, label_index in zip(file_names, logits):
            str_ = file_name + ' ' + labels[label_index] + '\n'
            f.write(str_)




  # run_graph(graph_cur, t, labels, FLAGS.input_layer, FLAGS.output_layer,
  #           FLAGS.num_top_predictions)

  # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
  # for tensor_name in tensor_name_list:
  #     print(tensor_name, '\n')
  #  and

  # with tf.Session() as sess:
  #     print(t.shape)



if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)