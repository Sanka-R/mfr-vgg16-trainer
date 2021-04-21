import cv2
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import img_to_array
# from tensorflow.python.keras.applications.inception_v3 import preprocess_input
# from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from keras_vggface.utils import preprocess_input
from recognition.config import IMG_SHAPE


class PairDataGenerator(object):
    """docstring for DataGenerator"""

    def __init__(self, batch_sz, dataset_base_folder, data_file, validation_split=0.2):
        self.batch_sz = batch_sz

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
        train_limit = int(len(references) * (1 - validation_split))
        self.train_references = references[:train_limit]
        self.val_references = references[train_limit:]

        self.train_probes = probes[:train_limit]
        self.val_probes = probes[train_limit:]

        self.train_true_labels = true_labels[:train_limit]
        self.val_true_labels = true_labels[train_limit:]

        self.samples_per_train = (len(self.train_references) / self.batch_sz) * self.batch_sz
        self.samples_per_val = (len(self.val_references) / self.batch_sz) * self.batch_sz
        # for inference assumed validation split is 0
        self.samples_per_inference = (len(self.train_references) / self.batch_sz) * self.batch_sz

        self.cur_train_index = 0
        self.cur_val_index = 0
        self.inference_index = 0

    def get_training_steps(self):
        return int(self.samples_per_train / self.batch_sz)

    def get_validation_steps(self):
        return int(self.samples_per_val / self.batch_sz)

    def __image_margin(self, image):
        shape = IMG_SHAPE
        background = np.zeros(shape, dtype=np.uint8)
        image_shape = image.shape

        r = int((shape[0] - image_shape[0]) / 2)
        c = int((shape[1] - image_shape[1]) / 2)

        background[r:r + image_shape[0], c:c + image_shape[1], :] = image

        return background

    def load_preprocessed_image(self, np_path):

        return None

    def get_scaled_image(self, image):
        image_shape = image.shape
        required_shape = IMG_SHAPE

        if image_shape[0] > image_shape[1]:
            resize_shape = (required_shape[0], int((required_shape[0] * image_shape[1]) / (image_shape[0])))

        else:
            resize_shape = (int((required_shape[0] * image_shape[0]) / (image_shape[1])), required_shape[1])

        resized_image = cv2.resize(image, resize_shape)
        return self.__image_margin(resized_image)

    def __preprocess_image(self, image):
        shape = IMG_SHAPE
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def __get_images(self, references, probes, true_labels):
        reference_images = []
        probe_images = []
        for idx, reference in enumerate(references):
            reference_image = cv2.cvtColor(cv2.imread(reference), cv2.COLOR_BGR2RGB)
            reference_image = self.get_scaled_image(reference_image)

            probe_image = cv2.cvtColor(cv2.imread(probes[idx]), cv2.COLOR_BGR2RGB)
            probe_image = self.get_scaled_image(probe_image)

            reference_images.append(reference_image)
            probe_images.append(probe_image)

        reference_images = np.array(reference_images)
        probe_images = np.array(probe_images)

        return [reference_images, probe_images], np.array(true_labels).astype(np.float32)

    def get_inference_images(self):
        status = True
        if self.inference_index >= self.samples_per_inference:
            status = False

        if status:
            x, y = self.__get_images(self.train_references[self.inference_index:self.inference_index + self.batch_sz],
                                     self.train_probes[self.inference_index:self.inference_index + self.batch_sz],
                                     self.train_true_labels[self.inference_index:self.inference_index + self.batch_sz])
            self.inference_index += self.batch_sz
            return x, y
        return None, None

    def next_train(self):
        while True:
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_inference:
                self.cur_train_index = 0

            yield self.__get_images(self.train_references[self.cur_train_index:self.cur_train_index + self.batch_sz],
                                    self.train_probes[self.cur_train_index:self.cur_train_index + self.batch_sz],
                                    self.train_true_labels[self.cur_train_index:self.cur_train_index + self.batch_sz])

    def next_val(self):
        while True:
            self.cur_val_index += self.batch_sz
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0

            yield self.__get_images(self.val_references[self.cur_val_index:self.cur_val_index + self.batch_sz],
                                    self.val_probes[self.cur_val_index:self.cur_val_index + self.batch_sz],
                                    self.val_true_labels[self.cur_val_index:self.cur_val_index + self.batch_sz])


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, ref_file_names, probe_file_names, labels, batch_size):
        self.batch_size = batch_size
        self.ref_file_names = ref_file_names
        self.probe_file_names = probe_file_names
        self.labels = labels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ref_file_names) / self.batch_size))

    def __getitem__(self, index):
        # print(index)
        'Generate one batch of data'
        # Generate indexes of the batch
        ref_curr_batch = self.ref_file_names[index * self.batch_size:(index + 1) * self.batch_size]
        probe_curr_batch = self.probe_file_names[index * self.batch_size:(index + 1) * self.batch_size]
        labels_curr_batch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__get_images(ref_curr_batch, probe_curr_batch, labels_curr_batch)
        # print(X[0].shape)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __image_margin(self, image):
        shape = IMG_SHAPE
        background = np.zeros(shape, dtype=np.uint8)
        image_shape = image.shape

        r = int((shape[0] - image_shape[0]) / 2)
        c = int((shape[1] - image_shape[1]) / 2)

        background[r:r + image_shape[0], c:c + image_shape[1], :] = image

        return background

    def __get_images(self, references, probes, labels):
        reference_images = []
        probe_images = []

        for idx, reference in enumerate(references):
            image = cv2.cvtColor(cv2.imread(reference), cv2.COLOR_BGR2RGB)
            image = self.__image_margin(image)
            # image = self.__preprocess_image(image)
            image = img_to_array(image)
            image = preprocess_input(image, version=1)
            reference_images.append(image)

        for probe in probes:
            image = cv2.cvtColor(cv2.imread(probe), cv2.COLOR_BGR2RGB)
            image = self.__image_margin(image)
            # image = self.__preprocess_image(image)
            image = img_to_array(image)
            image = preprocess_input(image, version=1)
            probe_images.append(image)
        reference_images = np.array(reference_images)
        probe_images = np.array(probe_images)
        if len(reference_images) == 0:
            print(references)

        return [reference_images, probe_images], np.array(labels).astype(np.float32)


class DataSplitter:

    def __init__(self, batch_sz, dataset_base_folder, data_file, validation_split=0.2):
        self.batch_sz = batch_sz
        references = []
        probes = []
        true_labels = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                values = line.split(" ")

                if i > 2500:
                    break

                reference = dataset_base_folder + values[0][2:]
                probe = dataset_base_folder + values[1][2:]
                label = int(values[-1].strip())

                references.append(reference)
                probes.append(probe)
                true_labels.append(label)
                i += 1

        train_limit = int(len(references) * (1 - validation_split))
        self.train_references = references[:train_limit]
        self.val_references = references[train_limit:]

        self.train_probes = probes[:train_limit]
        self.val_probes = probes[train_limit:]

        self.train_true_labels = true_labels[:train_limit]
        self.val_true_labels = true_labels[train_limit:]

    def get_training_details(self):
        return self.train_references, self.train_probes, self.train_true_labels, self.batch_sz

    def get_validation_details(self):
        return self.val_references, self.val_probes, self.val_true_labels, self.batch_sz
