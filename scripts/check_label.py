import codecs
from absl import flags, app

flags.DEFINE_string('input_file', None, '')
FLAGS = flags.FLAGS


def main(_):
    label_set = set(['N/A', 'PER', 'ORG', "LOC", "TIME"])
    for i, line in enumerate(open(FLAGS.input_file, 'r')):
        terms = line.strip('\n').split('\t')
        if terms[1] not in label_set:
            print('{}\t{}'.format(i, terms[1]))


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    app.run(main)
