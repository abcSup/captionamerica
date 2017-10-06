import collections
import os
import json
import random
import pickle

import h5py
import numpy as np
from tqdm import tqdm
from scipy.misc import imread, imresize

from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input

def isFilename(word):
	try:
		return word.encode('ascii')
	except UnicodeEncodeError:
		return False

train_dir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
val_dir =  ''
data_dir = './data/'
train_token_fname = 'train_token.txt'
val_token_fname = 'val_token.txt'

token_fname = train_token_fname
img_dir = train_dir
output_h5 = 'train_data.h5'

img_to_caps = {}
words = []
longest_s = 0
with open(os.path.join(data_dir, token_fname), encoding="utf-8") as file:
    for l in file:
        if isFilename(l):
            temp_imgname = l.replace(' ', '').strip()
            img_to_caps[temp_imgname] = []
        else:
            words.extend(l.split())
            img_to_caps[temp_imgname].append(l)

            len_s = len(l.split())
            longest_s = len_s if longest_s < len_s else longest_s

print('No. of images ', len(img_to_caps))
print('No. of words ', len(words))
print('No. of vocab ', len(collections.Counter(words)))
print('Longest captions ', longest_s)

def build_vocab(words, vocab_size):
    """Create vocabulary"""
    count = [['empty', -1], ['begin', -1], ['end', -1], ['unk', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 4))

    wtoi = dict()
    for word, _ in count:
        wtoi[word] = len(wtoi)
    #data = list()
    #unk_count = 0
    #for word in words:
    #    if word in dictionary:
    #        index = dictionary[word]
    #    else:
    #        index = 3  # 'unk'
    #        unk_count += 1
    #    data.append(index)
    #count[3][1] = unk_count
    itow = dict(zip(wtoi.values(), wtoi.keys()))
    return wtoi, itow

vocab_size = 10000
seq_len = 39 + 2
wtoi, itow = build_vocab(words, vocab_size)

del words

img_to_seqs = {}
for img_name, captions in img_to_caps.items():
    seqs = []
    for caption in captions:
        seq = [wtoi[w] for w in caption.split() if w in wtoi]
        seq.insert(0, wtoi['begin'])
        seq.insert(len(seq), wtoi['end'])
        seqs.append(seq)
    img_to_seqs[img_name] = seqs

del img_to_caps

def partialseqs_nextwords(img_name):
    seqs = img_to_seqs[img_name]
    input_seqs, target_seqs  = [], []
    for seq in seqs:
        input_seqs.append(seq[:-1])
        target_seqs.append(seq[1:])
    input_seqs = sequence.pad_sequences(input_seqs, maxlen=seq_len, padding='post')
    target_seqs = sequence.pad_sequences(input_seqs, maxlen=seq_len, padding='post')

    sample_weights = np.zeros([len(target_seqs), seq_len])
    sample_weights[target_seqs>0] = 1

    target_seqs = to_categorical(target_seqs, vocab_size)
    target_seqs = np.reshape(target_seqs, (len(input_seqs), seq_len, vocab_size))
    #target_seqs_1hot = np.zeros([len(target_seqs), seq_len, vocab_size], dtype=np.bool)
    #for i, seq in enumerate(target_seqs):
    #    for j, w in enumerate(seq):
    #        target_seqs_1hot [i, j, w] = 1

    print(target_seqs[0])
    print(input_seqs.shape, target_seqs.shape, sample_weights.shape)
    return input_seqs, target_seqs, sample_weights

def preprocess_img(img_name):
	"""Preprocess image for InceptionV3"""
	path = os.path.join(img_dir, img_name)
	img = imresize(imread(path), (299, 299)).astype(np.float32)
	img = preprocess_input(img)

	return img

img_names = [x for x in img_to_seqs.keys()]
N = len(img_names)
random.seed(123)
random.shuffle(img_names)

pickle.dump(wtoi, open("wtoi.p", 'wb'))
pickle.dump(itow, open("itow.p", 'wb'))

#f = h5py.File(os.path.join(data_dir, output_h5), 'w')
#img_set = f.create_dataset("images", (N, 299, 299, 3), dtype='float32')
#for i in tqdm(range(N)):
#    try:
#        img_set[i] = preprocess_img(img_names[i])
#    except:
#        print(img_names[i])
#f.close()
