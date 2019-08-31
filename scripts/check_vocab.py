import codecs


SPACES = set([c for c in u'\u0020\u00A0\u1680\u180E\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u200A\u200B\u202F\u205F\u3000\uFEFF'])

VOCAB_FILE = 'chinese_L-12_H-768_A-12/vocab.txt'

vocab = {}
for i, line in enumerate(open(VOCAB_FILE, 'r')):
    vocab[line.strip('\n')] = i

print('vocab size: {}'.format(len(vocab)))

for sym in ('[UNK]', '[PAD]', '[CLS]', '[SEP]', 'X'): 
    print('{}: {}'.format(sym, vocab.get(sym, -1)))
