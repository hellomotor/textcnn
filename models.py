import functools
import tensorflow as tf
from pathlib import Path


def model_fn_builder(document_max_len,
                     vocab_file,
                     label_list,
                     vocabulary_size,
                     embedding_size,
                     filter_sizes,
                     num_filters,
                     learning_rate,
                     unk_id):
    def model_fn(features, labels, mode, params):

        if isinstance(features, dict):
            features = features['words']

        # features = tf.Print(features, [features], message="[features]", summarize=80)
        vocab_words = tf.contrib.lookup.index_table_from_file(vocab_file, default_value=unk_id)
        num_class = len(label_list)
        feature_ids = vocab_words.lookup(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        keep_prob = tf.where(is_training, 0.5, 1.0)
        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            x_emb = tf.nn.embedding_lookup(embeddings, feature_ids)
            x_emb = tf.expand_dims(x_emb, -1)
        pooled_outputs = []
        for filter_size in filter_sizes:
            conv = tf.layers.conv2d(
                x_emb,
                filters=num_filters,
                kernel_size=[filter_size, embedding_size],
                strides=(1, 1),
                padding='VALID',
                activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[document_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
            pooled_outputs.append(pool)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters * len(filter_sizes)])
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob)

        with tf.name_scope("output"):
            logits = tf.layers.dense(h_drop, num_class, activation=None)
            predictions = tf.argmax(logits, -1, output_type=tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # batch_nums = tf.range(0, limit=logits.get_shape().as_list()[0])
            # indices = tf.stack((batch_nums, predictions), axis=1)
            # scores = tf.gather_nd(logits, indices)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
        else:
            vocab_labels = tf.contrib.lookup.index_table_from_tensor(tf.constant(label_list))
            label_ids = vocab_labels.lookup(labels)
            with tf.name_scope("loss"):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ids))
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    training_hooks=[])
            else:
                eval_metric_ops = {
                    'precision': tf.metrics.precision(label_ids, predictions),
                    'accuracy': tf.metrics.accuracy(label_ids, predictions),
                    'recall': tf.metrics.recall(label_ids, predictions)
                }
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=eval_metric_ops
                )
        return output_spec

    return model_fn


def input_fn_builder(FLAGS, split, document_max_len, shuffle_and_repeat=False):
    label_index, phrase_index = 1, 4

    def parse_fn(phrase, label):
        tokens = [ch.encode('utf8') for ch in phrase]
        if len(tokens) > document_max_len:
            tokens = tokens[:document_max_len]
        else:
            tokens.extend(['[PAD]'] * (document_max_len - len(tokens)))
        return tokens, label

    def generator_fn(path):
        with path.open(encoding='utf8', mode='r') as f:
            for line in f:
                tokens = line.strip('\n').split('\t')
                yield parse_fn(tokens[phrase_index], tokens[label_index])

    def input_fn(params):
        shapes = ([document_max_len], ())
        types = (tf.string, tf.string)
        defaults = ('[PAD]', 'N/A')

        input_path = Path(FLAGS.data_dir, '{}.txt'.format(split))
        dataset = tf.data.Dataset.from_generator(
            functools.partial(generator_fn, input_path),
            output_shapes=shapes, output_types=types)
        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['num_epochs'])
        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20), shapes, defaults, drop_remainder=True)
                   .prefetch(1))
        return dataset

    return input_fn
