import numpy as np
import cv2
import xml.etree.ElementTree as ET
import argparse
import os
import re
import nltk
import tensorflow as tf
from imutils import paths


def parse_images(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.resize(image, size=[224, 224])
    image = image / 255 - 0.5
    return image

def DatasetSSL(image_path, BUFFER_SIZE=500, BATCH_SIZE=32):
    train_images = list(paths.list_images(image_path))

    train_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_ds = (
        train_ds
        .map(parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(1024)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return train_ds