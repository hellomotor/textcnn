"""Export model as a saved_model"""
from pathlib import Path
import json
from models import model_fn_builder
import tensorflow as tf
from absl import flags, app

flags.DEFINE_string("model_dir", None, "")
FLAGS = flags.FLAGS


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    receiver_tensors = {'words': words}
    features = {'words': words}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main(_):
    with Path(FLAGS.model_dir, 'params.json').open() as f:
        params = json.load(f)
    model_fn = model_fn_builder(
        document_max_len=params['document_max_len'],
        vocab_file=params['vocab_file'],
        label_list=params['label_list'],
        vocabulary_size=params['vocabulary_size'],
        embedding_size=params['embedding_size'],
        filter_sizes=params['filter_sizes'],
        num_filters=params['num_filters'],
        learning_rate=params['learning_rate'],
        unk_id=params['unk_id'])
    estimator = tf.estimator.Estimator(model_fn, FLAGS.model_dir, params=params)
    estimator.export_saved_model('saved_model', serving_input_receiver_fn)


if __name__ == '__main__':
    flags.mark_flag_as_required('model_dir')
    app.run(main)
