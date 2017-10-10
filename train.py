from os.path import join
import numpy as np
import h5py

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from configuration import Config
from model import get_model
from data_generator import DataGenerator
from data_generator import CustomSequence

num_train_epoch = 60
num_finetune_epoch = 200
#data_dir = './data'
#h5file = join(data_dir, 'train_data.h5') 
#data = h5py.File(h5file, 'r')
#seq_generator = CustomSequence(data, batch_size)

config = Config()
model = get_model(config)
data = DataGenerator(config)

checkpoint = ModelCheckpoint('./checkpoint/_weight.hdf5',
				monitor='loss',
				mode='min',
				save_best_only=True,
				verbose=1)

tensorboard = TensorBoard(log_dir='./logs',
				histogram_freq=0,
				write_graph=True,
				write_images=False)

model.fit_generator(data.generator(), config.num_batch,
                    epochs=num_train_epoch,
                    initial_epoch=0,
                    callbacks=[tensorboard, checkpoint],
                    workers=1,
                    use_multiprocessing=False)
model.save_weights('./checkpoint/_weight.hdf5')

# Fine-tuning
config.fine_tune = True
model = get_model(config)

checkpoint = ModelCheckpoint('./checkpoint/train_weight-{epoch:03d}.hdf5',
				monitor='loss',
				mode='min',
				save_best_only=True,
				verbose=1)

earlystop = EarlyStopping(monitor='loss',
                          patience=10,
                          verbose=0,
                          mode='min')

model.load_weights('./checkpoint/_weight.hdf5')
model.fit_generator(data.generator(), config.num_batch,
                    epochs=num_finetune_epoch,
                    initial_epoch=num_train_epoch,
                    callbacks=[checkpoint, tensorboard, earlystop],
                    workers=1,
                    use_multiprocessing=False)

