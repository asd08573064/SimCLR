import numpy as np
import cv2
import xml.etree.ElementTree as ET
import argparse
import os
import re
import nltk
import tensorflow as tf
from imutils import paths



def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def mapping(image_path, train_captions):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0 - 0.5
    train_captions = tf.dtypes.cast(train_captions, tf.float32)
    return img, train_captions


def parse_images(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.resize(image, size=[224, 224])
    image = image / 255 - 0.5
    return image

def tokenize(new_report_list):
    top_k = 1000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+-/:;=?@[\]^_`{|}~')
    tokenizer.fit_on_texts(new_report_list)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(new_report_list)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    return cap_vector

def DatasetTransformer(BUFFER_SIZE=500, BATCH_SIZE=32):
    nltk.download('punkt')
    file_list = os.listdir('./IUXR_report')
    file_list.sort()
    image_path = []
    caption_list = []
    new_report_list = []
    new_image_name_list = []
    for file in file_list:
        caption_string = ''
        tree = ET.parse('./IUXR_report/' + file)
        root = tree.getroot()
        if root.find('parentImage') == None:
            continue
        else:
            image_path.append('IUXR_png/'+root.find('parentImage').attrib['id']+'.png')
        for data in root.iter('AbstractText'):
            label = data.attrib['Label']
            if label == 'FINDINGS' and data.text != None:
                caption_string+=data.text
            if label == 'IMPRESSION' and data.text != None:
                caption_string+=data.text
        caption_list.append(caption_string)
    
    for idx, report in enumerate(caption_list):
        new_report = report.lower().replace("..", ".")
        new_report = new_report.replace("'", "")
        new_sentences = []
        for sentence in nltk.tokenize.sent_tokenize(new_report):
            new_sentence = sentence.replace("/", " / ")
            if "xxxx" not in sentence and not hasNumbers(sentence):
                new_sentences.append(sentence)
        new_report = '<start> ' + " ".join(new_sentences) + ' <end>'
        if len(new_report) > 0:
            new_report_list.append(new_report)
            new_image_name_list.append(image_path[idx])

    cap_vector = tokenize(new_report_list)
    dataset_train = tf.data.Dataset.from_tensor_slices((new_image_name_list[0:3028], cap_vector[0:3028])).map(mapping).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_val = tf.data.Dataset.from_tensor_slices((new_image_name_list[3027:3827], cap_vector[3027:3827])).map(mapping).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset_train, dataset_val

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