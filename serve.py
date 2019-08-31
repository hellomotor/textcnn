"""Reload and serve a saved model"""
import sys
from pathlib import Path
from tensorflow.contrib import predictor
from absl import flags, app

flags.DEFINE_string("export_dir", None, "")
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
    subdirs = [x for x in Path(FLAGS.export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    while True:
        sys.stdout.write('=> ')
        text = sys.stdin.readline().strip('\n')
        if not text: break
        words = [q2b_dic.get(w, w).encode('utf8') for w in text.decode('utf8')]
        words.extend(['[PAD]'] * (document_max_len - len(words)))
        predictions = predict_fn({'words': [words]})
        i = predictions['output'][0]
        print(label_list[i])


if __name__ == '__main__':
    flags.mark_flag_as_required('export_dir')
    app.run(main)
