"""Reload and serve a saved model"""
import sys
from pathlib import Path
from tensorflow.contrib import predictor
from absl import flags, app

flags.DEFINE_string("export_dir", None, "")
flags.DEFINE_string("input_file", None, "")
FLAGS = flags.FLAGS


def load_dict(path, encoding='utf8', key_index=0, value_index=1):
    result = {}
    with Path(path).open('r', encoding=encoding) as f:
        for line in f:
            terms = line.strip('\n').split('\t')
            if len(terms) == 2:
                result[terms[key_index]] = terms[value_index]
    return result


def main(_):
    document_max_len = 60
    label_list = ['N/A', 'PER', 'ORG', "LOC", "TIME"]
    q2b_dic = load_dict('./q2b.dic')
    subdirs = [x for x in Path(FLAGS.export_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    for line in open(FLAGS.input_file):
        cols = line.strip('\n').split('\t')
        entity = cols[3].decode('utf8')
        words = [q2b_dic.get(w, w).encode('utf8') for w in entity]
        words.extend(['[PAD]'] * (document_max_len - len(words)))
        predictions = predict_fn({'words': [words]})
        pred_label = label_list[predictions['output'][0]]
        if pred_label != cols[1]:
            print('{} | {} | label: {} | predict: {}'.format(cols[-1], cols[3], cols[1], pred_label))


if __name__ == '__main__':
    flags.mark_flag_as_required('export_dir')
    flags.mark_flag_as_required('input_file')
    app.run(main)
