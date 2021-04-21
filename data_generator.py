import cv2
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import img_to_array
# from tensorflow.python.keras.applications.inception_v3 import preprocess_input
# from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from keras_vggface.utils import preprocess_input
from recognition.config import IMG_SHAPE

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

    def get_scaled_image(self, image):
        image_shape = image.shape
        required_shape = IMG_SHAPE

        if image_shape[0] > image_shape[1]:
            resize_shape = (required_shape[0], int((required_shape[0] * image_shape[1]) / (image_shape[0])))

        else:
            resize_shape = (int((required_shape[0] * image_shape[0]) / (image_shape[1])), required_shape[1])

        resized_image = cv2.resize(image, resize_shape)
        return self.__image_margin(resized_image)

    def __get_images(self, references, probes, labels):
        reference_images = []
        probe_images = []

        for idx, reference in enumerate(references):
            image = cv2.cvtColor(cv2.imread(reference), cv2.COLOR_BGR2RGB)
            image = self.get_scaled_image(image)
            # image = self.__preprocess_image(image)
            image = img_to_array(image)
            image = preprocess_input(image, version=2)
            reference_images.append(image)

        for probe in probes:
            image = cv2.cvtColor(cv2.imread(probe), cv2.COLOR_BGR2RGB)
            image = self.get_scaled_image(image)
            # image = self.__preprocess_image(image)
            image = img_to_array(image)
            image = preprocess_input(image, version=2)
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
