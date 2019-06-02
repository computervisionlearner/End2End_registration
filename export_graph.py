import tensorflow as tf

import model_reg as model


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', 'ckpt29_7', 'checkpoints directory path')
tf.flags.DEFINE_string('model', 'SoftRegNet.pb', 'image matching model name, default: model.pb')
tf.flags.DEFINE_integer('step', '302000', '')
tf.flags.DEFINE_integer('image_size', '64', 'image size, default: 256')

batch_size = None



def export_graph(model_name):
  graph = tf.Graph()

  with graph.as_default():
    

    input_image = tf.placeholder(tf.float32, shape=[batch_size, FLAGS.image_size, FLAGS.image_size, 1], name='input_image')
    flag_pl = tf.cast(False, tf.bool)
    _, features1 = model.get_features(input_image, 128, flag_pl, reuse = False)
   
    output_features = tf.identity(features1, name='output')
    restore_saver = tf.train.Saver()


  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver_path = "{}/model.ckpt-{}".format(FLAGS.checkpoint_dir, FLAGS.step)
    restore_saver.restore(sess, saver_path)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [output_features.op.name])

    tf.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)


def main(unused_argv):
  print('Export XtoY model...')
  export_graph(FLAGS.model)


if __name__ == '__main__':
  tf.app.run()
