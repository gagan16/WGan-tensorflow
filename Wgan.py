from __future__ import division
import tensorflow as tf
import time
from glob import glob
import imageio
from skimage.transform import resize
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import keras.backend as K

class Wgan(object):
    def __init__(self,sess,args):
        self.model_name = "Dcgan"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir=args.result_dir


        self.log_dir = args.log_dir
        self.epoch=args.epoch
        self.batch_size=args.batch_size
        self.image_size=args.img_size
        self.learning_rate=args.learning_rate
        self.print_freq=args.print_freq
        self.c_dim = 1
        self.channel=3
        self.z_dim=128
        self.image_shape=[self.image_size,self.image_size,self.channel]

        print()

        print("##### Information #####")
        print("# Dcgan")
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)


        print("# Image size : ", self.image_size)
        print("# learning rate : ", self.learning_rate)

        print()

    def wasserstein_loss(self,y_true, y_pred):
        return K.mean(y_true * y_pred)

    def construct_critic(self,image_shape):

        # weights need to be initialized with close values near zero to avoid
        # clipping
        weights_initializer = RandomNormal(mean=0., stddev=0.01)

        critic = Sequential()
        critic.add(Conv2D(filters=64, kernel_size=(5, 5),
                          strides=(2, 2), padding='same',
                          data_format='channels_last',
                          kernel_initializer=weights_initializer,
                          input_shape=(image_shape)))
        critic.add(LeakyReLU(0.2))

        critic.add(Conv2D(filters=128, kernel_size=(5, 5),
                          strides=(2, 2), padding='same',
                          data_format='channels_last',
                          kernel_initializer=weights_initializer))
        critic.add(BatchNormalization(momentum=0.5))
        critic.add(LeakyReLU(0.2))

        critic.add(Conv2D(filters=256, kernel_size=(5, 5),
                          strides=(2, 2), padding='same',
                          data_format='channels_last',
                          kernel_initializer=weights_initializer))
        critic.add(BatchNormalization(momentum=0.5))
        critic.add(LeakyReLU(0.2))

        critic.add(Conv2D(filters=512, kernel_size=(5, 5),
                          strides=(2, 2), padding='same',
                          data_format='channels_last',
                          kernel_initializer=weights_initializer))
        critic.add(BatchNormalization(momentum=0.5))
        critic.add(LeakyReLU(0.2))

        critic.add(Flatten())

        # We output two layers, one witch predicts the class and other that
        # tries to figure if image is fake or not
        critic.add(Dense(units=1, activation=None))
        optimizer = RMSprop(lr=0.00005)
        critic.compile(loss=self.wasserstein_loss,
                       optimizer=optimizer,
                       metrics=None)

        return critic

    def construct_generator(self):

        weights_initializer = RandomNormal(mean=0., stddev=0.01)

        generator = Sequential()

        generator.add(Dense(units=4 * 4 * 512,
                            kernel_initializer=weights_initializer,
                            input_shape=(1, 1, 100)))
        generator.add(Reshape(target_shape=(4, 4, 512)))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=weights_initializer))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=weights_initializer))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=weights_initializer))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=32, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=weights_initializer))
        generator.add(Activation('tanh'))

        optimizer = RMSprop(lr=0.00005)
        generator.compile(loss=self.wasserstein_loss,
                          optimizer=optimizer,
                          metrics=None)

        return generator

    def build_model(self):

        # Build the adversarial model that consists in the generator output
        # connected to the critic
        generator = self.construct_generator()
        generator.summary()

        critic = self.construct_critic(self.image_shape)
        critic.summary()

        gan = Sequential()
        # Only false for the adversarial model
        critic.trainable = False
        gan.add(generator)
        gan.add(critic)
        # gan.summary()
        optimizer = RMSprop(lr=0.00005)
        gan.compile(loss=self.wasserstein_loss,
                    optimizer=optimizer,
                    metrics=None)

        #Loading the dataset and converting images to feed to discrminator

        path = glob('dataset/'+self.dataset_name)
        batch = np.random.choice(path, self.batch_size)
        imgs = []
        for img in batch:
            img = self.imread(img)
            img = resize(img, self.image_shape)
            imgs.append(img)

        number_of_batches = int(len(path) / self.batch_size)


        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        #saving checkpoints
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / number_of_batches)
            start_batch_id = checkpoint_counter - start_epoch * number_of_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS ", counter)
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()

        # Tensorboard log variable
        summary_writer = tf.summary.FileWriter('logs/WGAN')

        # Create the plot that will show the losses
        plt.ion()

        # Variables used for loss saving
        generator_iterations = 0
        d_loss = 0
        d_real = 0
        d_fake = 0
        g_loss = 0

        # Let's train the WGAN for n epochs
        for epoch in range(start_epoch,self.epoch):

            for batch_number in range(start_batch_id, number_of_batches):
                start_time = time.time()

                # Just like the v2 version of paper, in the first 25 epochs, the critic
                # is updated 100 times for each generator update. Occasionally (each 500
                # epochs this is repeated). In the other epochs the default value is 5
                if generator_iterations < 25 or (generator_iterations + 1) % 500 == 0:
                    critic_iterations = 100
                else:
                    critic_iterations = 5

                # Update the critic a number of critic iterations
                for critic_iteration in range(critic_iterations):

                    if batch_number > number_of_batches:
                        break

                    real_images = np.array(imgs) / 127.5 - 1.
                    # batch_number += 1

                    # The last batch is smaller than the other ones, so we need to
                    # take that into account
                    current_batch_size = real_images.shape[0]

                    # Generate noise
                    noise = np.random.normal(0, 1,
                                             size=(current_batch_size,) + (1, 1, 100))

                    # Generate images
                    generated_images = generator.predict(noise)

                    # Add some noise to the labels that will be fed to the critic
                    real_y = np.ones(current_batch_size)
                    fake_y = np.ones(current_batch_size) * -1

                    # Let's train the critic
                    critic.trainable = True

                    # Clip the weights to small numbers near zero
                    for layer in critic.layers:
                        weights = layer.get_weights()
                        weights = [np.clip(w, -0.01, 0.01) for w in weights]
                        layer.set_weights(weights)

                    d_real = critic.train_on_batch(real_images, real_y)
                    d_fake = critic.train_on_batch(generated_images, fake_y)

                    d_loss = d_real - d_fake

                # numpy array that will store the losses to be passed to tensorboard
                losses = np.empty(shape=1)
                losses = np.append(losses, d_real)
                losses = np.append(losses, d_fake)

                # Update the generator
                critic.trainable = False

                noise = np.random.normal(0, 1,
                                         size=(current_batch_size,) + (1, 1, 100))

                # We try to mislead the critic by giving the opposite labels
                fake_y = np.ones(current_batch_size)
                g_loss = gan.train_on_batch(noise, fake_y)

                losses = np.append(losses, g_loss)

                # Each 100 generator iterations show and save images
                if ((generator_iterations + 1) % 1 == 0):
                    noise = np.random.normal(0, 1, size=(64,) + (1, 1, 100))
                    generated_images = generator.predict(noise)
                    # save_generated_images(generated_images, generator_iterations)
                    self.sample_images(generated_images, epoch, generator_iterations)

                # Update tensorboard plots
                self.write_to_tensorboard(generator_iterations, summary_writer, losses)

                time_elapsed = time.time() - start_time
                counter += 1
                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f - %f s'
                      % (epoch, self.epoch, batch_number, number_of_batches, generator_iterations,
                         d_loss, g_loss, d_real, d_fake, time_elapsed))

                generator_iterations += 1

            if (epoch + 1) % 5 == 0:
                critic.trainable = True
                generator.save('models/generator_epoch' + str(epoch) + '.hdf5')
                critic.save('models/critic_epoch' + str(epoch) + '.hdf5')

            start_batch_id = 0
            print(counter)
            self.save(self.checkpoint_dir, counter)
            # self.visualize_results(epoch)
        print("main counter", counter)
        self.save(self.checkpoint_dir, counter)


    def imread(self,path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)

    def sample_images(self,generated_images, epoch, batch_i):
        os.makedirs(self.result_dir+'/'+self.dataset_name, exist_ok=True)


        # Translate images to the other domain
        fake_B = generated_images

        fake_B = 0.5 * fake_B + 0.5

        imageio.imwrite(self.result_dir+'/'+self.dataset_name+'/%d_Fake%d.png' % (epoch, batch_i), fake_B[0])

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def write_to_tensorboard(self,generator_step, summary_writer,
                             losses):

        summary = tf.Summary()

        value = summary.value.add()
        value.simple_value = losses[1]
        value.tag = 'Critic Real Loss'

        value = summary.value.add()
        value.simple_value = losses[2]
        value.tag = 'Critic Fake Loss'

        value = summary.value.add()
        value.simple_value = losses[3]
        value.tag = 'Generator Loss'

        value = summary.value.add()
        value.simple_value = losses[1] - losses[2]
        value.tag = 'Critic Loss (D_real - D_fake)'

        value = summary.value.add()
        value.simple_value = losses[1] + losses[2]
        value.tag = 'Critic Loss (D_fake + D_real)'

        summary_writer.add_summary(summary, generator_step)
        summary_writer.flush()

