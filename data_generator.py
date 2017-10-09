from os.path import join, exists
import pickle
import random

import numpy as np
import h5py
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input

class DataGenerator(object):

    def __init__(self, batch_size, num_steps):
        self.data_dir = './data/'
        self.img_dir = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
        self.wtoi = pickle.load(open(join(self.data_dir, 'wtoi.p'), 'rb'))
        self.itow = pickle.load(open(join(self.data_dir, 'itow.p'), 'rb'))
        self.imglist_file = join(self.data_dir, 'img_names.p')
        self.h5file = join(self.data_dir, 'train_data.h5') 

        self.vocab_size = 10000
        self.seq_len = 41 - 1
        self.batch_size = batch_size
        self.num_steps = num_steps

    def _preprocess_caps(self, caps, seq_len):
        input_seqs, target_seqs = [], []
        for seq in caps:
            input_seqs.append(seq[:-1])
            target_seqs.append(seq[1:])

        input_seqs = np.asarray(input_seqs)
        target_seqs = np.asarray(target_seqs)

        sample_weights = np.zeros([len(target_seqs), self.seq_len])
        sample_weights[target_seqs>0] = 1

        target_seqs = to_categorical(target_seqs, self.vocab_size)
        target_seqs = np.reshape(target_seqs, (len(input_seqs), self.seq_len, self.vocab_size))

        return input_seqs, target_seqs, sample_weights

    def _preprocess_imgs(self, img_names):
        """Preprocess image for InceptionV3"""
        images = np.zeros((self.batch_size,299,299,3),dtype=np.float32)
        for i, img_name in enumerate(img_names):
            path = join(self.img_dir, img_name)
            try:
                img = imresize(imread(path), (299,299)).astype(np.float32)
                img = preprocess_input(img)
            except:
                print("GGWP for image: %s" % img_name)
                continue
            images[i] = img
        
        return images

    def generator(self):
        batch_size = self.batch_size
        num_steps = self.num_steps

        data = h5py.File(self.h5file, 'r')
        captions = data['caps']
        starts = data['caps_start_idx']
        ends = data['caps_end_idx']

        img_names = pickle.load(open(self.imglist_file, 'rb'))
        img_names = np.array(img_names)

        while(True):           
            idxs = np.arange(len(img_names))
            np.random.shuffle(idxs)

            for i in range(num_steps):
                batch_idxs = idxs[i*batch_size:(i+1)*batch_size]
                batch_idxs = np.sort(batch_idxs).tolist()

                img_idxs = img_names[batch_idxs]
                images = self._preprocess_imgs(img_idxs)

                start_idxs = starts[batch_idxs]
                end_idxs = ends[batch_idxs]

                cap_idxs = []
                for x in range(batch_size):
                    cap_idxs.append(random.randint(start_idxs[x], end_idxs[x]))
                caps = captions[cap_idxs]
                
                input_seqs, output_seqs, weights  = self._preprocess_caps(caps, self.seq_len)

                yield ([images, input_seqs], output_seqs, weights)

    def test(self):
        for data in self.generator():
            images = data[0][0]
            input_seqs = data[0][1]
            output_seqs = data[1]

            for i, img in enumerate(images):
                for w in input_seqs[i]:
                    print(self.itow[w], end='')
                print()
                for x in output_seqs[i]:
                    w = np.argmax(x)
                    print(self.itow[w], end='')
                print()
                plt.imshow(img)
                plt.show()
        
if __name__ == "__main__":
    data = DataGenerator(32, 1)
    data.test()
