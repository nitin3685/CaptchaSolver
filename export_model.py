import sys
# Path to TF Serving library
sys.path.insert(0, "/srv/serving")
from tensorflow_serving.session_bundle import exporter

# Save model in protobuf format
def export_model(sess, export_path, export_version, x, y):
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=x, scores_tensor=y)
    model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)
    print("Export Finished!")
