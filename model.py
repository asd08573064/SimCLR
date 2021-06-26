import matplotlib.pyplot as plt

from hyperparameters import *
import numpy as np
from tqdm import tqdm
from imutils import paths
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing



class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images

def get_augmenter(min_area, brightness, jitter):
    zoom_factor = 1.0 - tf.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(para_SSL['image_size'], para_SSL['image_size'], para_SSL['image_channels'])),
            preprocessing.RandomFlip("horizontal"),
            preprocessing.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            RandomColorAffine(brightness, jitter),
        ]
    )



class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = para_SSL['temperature']
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.encoder = tf.keras.applications.ResNet50(include_top=True,weights=None, classes=1024, classifier_activation=None)
        self.encoder.trainable = True
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(1024,)),
                layers.Dense(para_SSL['width'], activation="relu"),
                layers.Dense(para_SSL['width']),
            ],
            name="projection_head",
        )
    def call(self, x, training=True):
        if training:

            augmented_images_1 = self.contrastive_augmenter(x)
            augmented_images_2 = self.contrastive_augmenter(x)
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            project_1 =  self.projection_head(features_1)
            project_2 =  self.projection_head(features_2)
        
            return project_1, project_2
        else:
            return self.encoder(x)

        

