import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from model import *
from dataset import *
import time
import os
import argparse



def contrastive_loss(projections_1, projections_2, temperature):
    projections_1 = tf.math.l2_normalize(projections_1, axis=1)
    projections_2 = tf.math.l2_normalize(projections_2, axis=1)
    similarities = (
        tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
    )
    batch_size = tf.shape(projections_1)[0]
    contrastive_labels = tf.range(batch_size)
    
    loss_1_2 = keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarities, from_logits=True
    )
    loss_2_1 = keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarities), from_logits=True
    )
    return (loss_1_2 + loss_2_1) / 2

def train_step(data, SSL_model, optimizer, train_loss, temperature, epoch):
    images = data    
    with tf.GradientTape() as tape:
        project1, project2 = SSL_model(images)
        loss = contrastive_loss(project1, project2, temperature)
    
    gradients = tape.gradient(loss, SSL_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, SSL_model.trainable_variables))

    train_loss(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",
                        type=int,
                        nargs=1,
                        default=150,
                        )
    parser.add_argument("--temperature",
                        type=float,
                        nargs=1,
                        default=0.1,
                        )
    parser.add_argument("--image_path",
                        type=str,
                        nargs=1,
                        default='./IUXR_png/',
                        )
    parser.add_argument("--checkpoint_path",
                        type=str,
                        nargs=1,
                        default='./checkpoints/SSL_ResNet50',
                        )

    args = parser.parse_args()
    epochs = args.epoch
    temperature = args.temperature
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    SSL_model = ContrastiveModel()  
    train_ds=DatasetSSL(image_path)
    ckpt = tf.train.Checkpoint(SSL_model=SSL_model,
                           optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    for epoch in range(epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        for batch, data in enumerate(train_ds):
            train_step(data, SSL_model, optimizer, train_loss, temperature, epoch)
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')
        
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


    
