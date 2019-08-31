import json

import tensorflow as tf
from absl import flags, app
from pathlib import Path

from models import model_fn_builder, input_fn_builder

flags.DEFINE_string("data_dir", default=None, help="")
flags.DEFINE_string("output_dir", default=None, help="")
flags.DEFINE_string("vocab_file", default='./chinese_L-12_H-768_A-12/vocab.txt', help="")
flags.DEFINE_integer("document_max_len", default=60, help="")

FLAGS = flags.FLAGS


def main(_):
    vocab = {}
    for i, line in enumerate(open(FLAGS.vocab_file, 'r')):
        vocab[line.strip('\n')] = i

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=session_config
    )
    unk_id = vocab['[UNK]']
    label_list = ['N/A', 'PER', 'ORG', "LOC", "TIME"]
    params = {
        "document_max_len": FLAGS.document_max_len,
        "vocabulary_size": 21128,
        "unk_id": unk_id,
        "embedding_size": 300,
        "filter_sizes": [2, 3, 4],
        "num_filters": 32,
        "vocab_file": FLAGS.vocab_file,
        "label_list": label_list,
        "learning_rate": 1e-3,
        "buffer": 1500,
        "num_epochs": 10,
        "batch_size": 128
    }
    with Path(FLAGS.output_dir, 'params.json').open('w', encoding="utf8") as f:
        f.write(json.dumps(params, ensure_ascii=False).decode('utf8'))
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

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    # eval_input_fn = text_input_fn_builder('dev')
    # tokens, label = eval_input_fn(params).make_one_shot_iterator().get_next()
    # with tf.Session() as session:
    #     while True:
    #         batch_tokens, batch_labels = session.run([tokens, label])
    #         print(batch_labels)

    train_input_fn = input_fn_builder(
        FLAGS,
        'train',
        document_max_len=params['document_max_len'],
        shuffle_and_repeat=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=None,
                                        hooks=[])
    eval_input_fn = input_fn_builder(
        FLAGS,
        'dev',
        document_max_len=params['document_max_len'])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    app.run(main)
