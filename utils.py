# -*- coding:utf-8 -*-
import codecs
import os
import pickle
import click
from bert_lstm_crf.bert import tokenization
from bert_lstm_crf.models import InputExample, InputFeatures
import tensorflow as tf
import collections
from absl import flags, app, logging

__all__ = ['DataProcessor', 'NerProcessor', 'write_tokens', 'convert_single_example',
           'filed_based_convert_examples_to_features', 'file_based_input_fn_builder']

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="")
flags.DEFINE_string("output_dir", default=None, help="")
flags.DEFINE_string("vocab_file", default=None, help="")
flags.DEFINE_integer("max_seq_length", default=202, help="")
flags.DEFINE_integer("batch_size", default=64, help="")
flags.DEFINE_integer("num_train_epochs", default=10, help="")
flags.DEFINE_bool("do_lower_case", default=True, help="")


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        self.labels = ["NA", 'PER', 'ORG', "LOC", "TIME", "JOB_TITLE"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines


def write_tokens(tokens, output_dir, mode):
    """
    ???????????????????????????????????????
    ??????mode=test???????????????
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    """
    ???????????????????????????????????????????????????id, ???????????????id,??????????????????InputFeatures?????????
    :param ex_index: index
    :param example: ????????????
    :param label_list: ????????????
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    # 1?????????1?????????label??????index???
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # ??????label->index ???map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # ???????????????????????????????????????,????????????????????????BERT???vocab.txt????????????????????????WordPice???????????????????????????????????????????????????????????????????????????list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # ??????????????????else
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # ????????????
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 ????????????????????????????????????????????????????????????
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # ??????????????????CLS ??????
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS ????????????????????????????????????O ?????????????????????,????????????????????????????????????????????????????????????LCS ????????????
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # ????????????[SEP] ??????
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # ??????????????????(ntokens)?????????ID??????
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, ??????
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # ??????????????????????????????
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # ?????????????????????
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    # mode='test'??????????????????
    write_tokens(ntokens, output_dir, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    ??????????????????TF_Record ?????????????????????????????????
    :param examples:  ??????
    :param label_list:??????list
    :param max_seq_length: ?????????????????????????????????
    :param tokenizer: tokenizer ??????
    :param output_file: tf.record ????????????
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # ??????????????????
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ???????????????????????????,
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature ??????????????????????????????????????????
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # ?????????????????????CPU????????????????????????????????????????????????
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def main(_):
    processor = NerProcessor(FLAGS.output_dir)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    label_list = processor.get_labels()
    num_train_steps = int(
        len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)
    if num_train_steps < 1:
        raise AttributeError('training data is so small...')

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_examples))
    logging.info("  Batch size = %d", FLAGS.batch_size)
    logging.info("  Num steps = %d", num_train_steps)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    if not os.path.exists(train_file):
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, FLAGS.output_dir)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    # ???????????????????????????
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_examples))
    logging.info("  Batch size = %d", FLAGS.batch_size)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, FLAGS.output_dir)

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    logging.info("***** Running prediction*****")
    logging.info("  Num examples = %d", len(predict_examples))
    logging.info("  Batch size = %d", FLAGS.batch_size)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if not os.path.exists(predict_file):
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, FLAGS.output_dir, mode="test")


if __name__ == '__main__':
    app.run(main)

