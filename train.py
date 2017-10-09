import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard

from model import get_model
from data_generator import DataGenerator

nb_epoch = 200
batch_size = 32
num_samples = 32

model = get_model()
data = DataGenerator(batch_size, num_samples)

checkpoint = ModelCheckpoint('./checkpoint/train_weight-{epoch:03d}.hdf5',
				monitor='loss',
				verbose=1,
				save_best_only=True,
				mode='min')
tensorboard = TensorBoard(log_dir='./logs',
				histogram_freq=0,
				write_graph=True,
				write_images=False)

model.fit_generator(data.generator(), num_samples,
                    epochs=nb_epoch,
                    callbacks=[tensorboard, checkpoint],
                    workers=1,
                    use_multiprocessing=False)
