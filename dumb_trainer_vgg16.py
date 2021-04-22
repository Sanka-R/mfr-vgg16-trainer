from datetime import datetime

import tensorflow as tf

from git_trainer_vgg16.build_networks import build_siamese_network_vgg16
from git_trainer_vgg16.data_generator import DataGenerator
from recognition.config import *

dataset_base_folder = "D:\Sanka\Academic\FaceComp\\training_dataset\celebrity_dataset\\"
data_file = "../outputs/celeb_a_1_list.txt"

references = []
probes = []
true_labels = []
with open(data_file, 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        values = line.split(" ")

        reference = dataset_base_folder + values[0][2:]
        probe = dataset_base_folder + values[1][2:]
        label = int(values[-1].strip())

        references.append(reference)
        probes.append(probe)
        true_labels.append(label)
        i += 1

validation_split = 0.05
batch_sz = BATCH_SIZE

train_limit = int(len(references) * (1 - validation_split))
train_references = references[:train_limit]
val_references = references[train_limit:]

train_probes = probes[:train_limit]
val_probes = probes[train_limit:]

train_true_labels = true_labels[:train_limit]
val_true_labels = true_labels[train_limit:]

samples_per_train = (len(train_references) / batch_sz) * batch_sz
samples_per_val = (len(val_references) / batch_sz) * batch_sz

cur_train_index = 0
cur_val_index = 0


# latest = tf.train.latest_checkpoint("./outputs/")

model = build_siamese_network_vgg16(fine_tune_percentage=0.01)

# model.load_weights(latest)

training_generator = DataGenerator(train_references, train_probes, train_true_labels, batch_sz)
validation_generator = DataGenerator(val_references, val_probes, val_true_labels, batch_sz)

model_architecture = "VGG16_contrastive"
pretrained_weights = "VGGFACE2"
percentage_freezed = "5"

date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
checkpoint_filepath = f'./outputs/checkpoint_{model_architecture}_{pretrained_weights}_{percentage_freezed}_{date_time}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_freq='epoch',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(x=training_generator,
          validation_data=validation_generator, callbacks=[model_checkpoint_callback],
          epochs=5)

model_saving_name = f"{model_architecture}_{pretrained_weights}_{percentage_freezed}_{date_time}"
model.save(f"../outputs/model_{model_saving_name}.h5")
