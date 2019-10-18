import tensorflow as tf

if __name__ == '__main__':
    graph_path = 'mtcnn.pb'
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    graph = tf.get_default_graph()
    pnet = tf.placeholder(shape=(None, 224, 224, 3), dtype='float32', name='pnet/input')
    rnet = tf.placeholder(shape=(None, 24, 24, 3), dtype='float32', name='rnet/input')
    onet = tf.placeholder(shape=(None, 48, 48, 3), dtype='float32', name='onet/input')
    tf.import_graph_def(graph_def, name='',
                        input_map={"pnet/input:0": pnet, "rnet/input:0": rnet, "onet/input:0": onet})
    # tf.import_graph_def(graph_def, name='rnet/input', input_map={"rnet/input:0": rnet})
    # tf.import_graph_def(graph_def, name='onet/input', input_map={"onet/input:0": onet})
    tf.train.write_graph(graph, "G:/G_Downloads/tf-mtcnn-master/", "mtcnn_v1.pb", as_text=False)
