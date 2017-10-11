from os.path import join, basename
import hashlib
import json
import multiprocessing as mp
import sys

import jieba
from tqdm import tqdm

annotation_file = '/mnt/Users/abcSup/Downloads/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
num_refs = 100

def tokenize(img):
    file_name = img['image_id'].rstrip('.jpg')
    img_id = int(int(hashlib.sha256(file_name.encode('utf-8')).hexdigest(), 16) % sys.maxsize)
    words, caps, length = [], [], []
    for c in img['caption']:
        cap = list(jieba.cut(c, cut_all=False))
        caps.append(cap)

    return file_name, img_id, caps

def read_dataset():
    pool = mp.Pool(mp.cpu_count())

    x = {}
    x['annotations'] = []
    x['images'] = []
    x['info'] = {'contributor': 'abc', 'description': 'abc',\
                 'url': 'abc', 'version': '1', 'year': 2017}
    x['licenses'] = [{'url': 'abc'}]
    x['type'] = 'captions'

    with open(annotation_file, encoding="utf-8") as file:
        data = json.load(file)

    result = pool.imap(tokenize, [img for img in data[:num_refs]])

    counter = 1
    for r in tqdm(result):
        annotation = []
        image = []

        for cap in r[2]:
            a = {'id': counter, 'image_id': r[1]}
            a['caption'] = ' '.join(cap)
            x['annotations'].append(a)

            img = {'file_name': r[0], 'id': r[1]}
            x['images'].append(img)

            counter += 1

    json.dump(x, open('./data/reference.json', 'w'),
              sort_keys=True, indent=2)

read_dataset()
