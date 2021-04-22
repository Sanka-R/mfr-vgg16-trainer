import tensorflow as tf
import tensorflow.keras.backend as K
from keras_vggface.vggface import VGGFace
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *

from recognition.config import IMG_SHAPE

def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type
    # tf.print(preds)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))

    loss = 1 - K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss

def square_change(vectors):
    # unpack the vectors into separate lists
    [featsA, featsB] = vectors
    # compute the sum of squared distances between the vectors
    sq = K.square((featsA - featsB))

    return sq


def build_siamese_network_vgg16(fine_tune_percentage):
    input_a = tf.keras.layers.Input(shape=IMG_SHAPE)
    input_b = tf.keras.layers.Input(shape=IMG_SHAPE)

    sister_network = build_vgg16_sister_network(IMG_SHAPE, True, fine_tune_percentage)

    feature_a = sister_network(input_a)
    feature_b = sister_network(input_b)

    diff = tf.keras.layers.Lambda(square_change)([feature_a, feature_b])

    # concat = tf.keras.layers.concatenate([feature_a, feature_b])
    # fc1 = tf.keras.layers.Dense(256, activation="relu")(diff)
    # fc2 = tf.keras.layers.Dense(64, activation="relu")(fc1)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(diff)
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)

    optimizer = tf.keras.optimizers.SGD(lr=0.001)
    # optimizer = tf.keras.optimizers.RMSprop(lr=0.001)
    # model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    return model


def build_vgg16_sister_network(shape, fineTune=False, fine_tune_percentage=.30):
    # base_model = tf.keras.applications.VGG16(input_shape=shape, include_top=False, weights="imagenet", pooling="avg")
    base_model = VGGFace(include_top=False, input_shape=shape, pooling="avg")
    base_model.summary()
    if fineTune == False:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Fine-tune from this layer onwards
        fine_tune_at = 17
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    # last_layer = base_model.get_layer('pool5').output
    # x = Flatten(name='flatten')(last_layer)
    out = Dense(512, activation='relu', name='fc7')(base_model.output)
    model = Model(base_model.input, out)
    model.summary()
    return model
