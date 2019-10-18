import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = 'saved_model'
graph_pb = 'mtcnn_v1.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    input_layer = ["pnet/input:0", "rnet/input:0", "onet/input:0"]
    inp = g.get_tensor_by_name(input_layer[0])
    inp1 = g.get_tensor_by_name(input_layer[1])
    inp2 = g.get_tensor_by_name(input_layer[2])

    output_layer = ["pnet/prob1:0", "pnet/conv4-2/BiasAdd:0", "rnet/prob1:0",
                    "rnet/conv5-2/conv5-2:0", "onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0"]
    out = g.get_tensor_by_name(output_layer[0]);
    out1 = g.get_tensor_by_name(output_layer[1]);
    out2 = g.get_tensor_by_name(output_layer[2]);
    out3 = g.get_tensor_by_name(output_layer[3]);
    out4 = g.get_tensor_by_name(output_layer[4]);
    out5 = g.get_tensor_by_name(output_layer[5]);
    out6 = g.get_tensor_by_name(output_layer[6]);

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in1": inp, "in2": inp1, "in3": inp2}, {"out": out, "out1": out1, "out2": out2, "out3": out3, "out4": out4,
                                                     "out5": out5, "out6": out6})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
