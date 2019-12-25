from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model,load_model,Sequential
from keras.layers import Input,Dense,Conv2D,BatchNormalization,Add,LeakyReLU,Flatten,Lambda,Reshape,UpSampling2D,Cropping2D,Subtract, ReLU,Dropout
from keras.layers import ZeroPadding2D,Concatenate,Activation,Concatenate
from keras.activations import sigmoid
from keras import regularizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
import keras.backend as K
import pandas as pd
import numpy as np
from tqdm import trange
from time import sleep
import math
import argparse
from glob import glob
import pickle
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib
import multiprocessing
import os
import keras
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)
import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        shape = K.shape(inputs[0])
        alpha = K.random_uniform((shape[0], 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class Encoder_Disentagled():
    def __init__(self):
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 1024

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.encoder = self.build_encoder()
        self.disengtangled_encoder = self.build_encoder()
        self.siamese_net = self.build_siamese()

        # self.encoder.trainble = False
        # self.encoder_target.trainble = False

        ##data
        path = '/mnt/hieuck/dataset/celebA/identity_CelebA.txt'
        root = '/mnt/hieuck/dataset/celebA/img_align_celeba'
        self.paths = pd.read_csv(path, sep = ' ', header = None)

        self.paths[2] = self.paths[0].apply(lambda x: os.path.join(root,x))

        self.groups = self.paths.groupby(1).groups
        self.index = [k for k in self.groups]
        self.batch_size = 2




        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------
        #
        # # Freeze generator's layers while training critic
        # self.encoder.trainable = False

        # Image input (real sample)
        source_img = Input(shape=self.img_shape)
        target_img = Input(shape=self.img_shape)

        #Embedding of source and target
        source_emb = self.encoder(source_img)
        target_emb = self.encoder(target_img)

        # # Construct weighted average between real and fake emb
        # interpolated_emb = RandomWeightedAverage()([source_emb, target_emb])



        # Discriminator determines validity of the real and fake images
        valid = self.siamese_net([source_emb,target_emb])


        # # Determine validity of weighted sample
        # validity_interpolated_source = self.critic([interpolated_emb,source_emb])
        # validity_interpolated_target = self.critic([interpolated_emb,target_emb])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        # partial_gp_loss = partial(self.gradient_penalty_loss,
        #                   averaged_samples=interpolated_emb)
        # partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.siamese_model = Model(inputs=[source_img, target_img],
                            outputs=[valid])
        self.siamese_model.compile(loss='binary_crossentropy',
                                        optimizer=optimizer)

        #-------------------------------
        # Construct Computational Graph
        #         for Disentangled feats
        #-------------------------------

        self.dis_siamese_net = self.build_siamese()
        self.disengtangled_encoder.trainable= False

        dis_source_img = Input(shape=self.img_shape)
        dis_target_img = Input(shape=self.img_shape)

        #Embedding of source and target
        dis_source_emb = self.disengtangled_encoder(dis_source_img)
        dis_target_emb= self.disengtangled_encoder(dis_target_img)

        # # Construct weighted average between real and fake emb
        # interpolated_emb_adv = RandomWeightedAverage()([source_emb_adv, target_emb_adv])


        # Discriminator determines validity of the real and fake images
        dis_valid = self.dis_siamese_net([dis_source_emb,dis_target_emb])



        self.dis_siamese_model = Model(inputs=[dis_source_img, dis_target_img],
                            outputs=[dis_valid])
        self.dis_siamese_model.compile(loss='binary_crossentropy',
                                        optimizer=optimizer)
        #-------------------------------
        # Construct Computational Graph
        #         for Adv Disentangled feats
        #-------------------------------

        self.dis_siamese_net.trainable =False
        self.disengtangled_encoder.trainable = True
        source_img_adv = Input(shape=self.img_shape)
        target_img_adv = Input(shape=self.img_shape)

        #Embedding of source and target
        source_emb_adv = self.disengtangled_encoder(source_img_adv)
        target_emb_adv = self.disengtangled_encoder(target_img_adv)

        # # Construct weighted average between real and fake emb
        # interpolated_emb_adv = RandomWeightedAverage()([source_emb_adv, target_emb_adv])


        # Discriminator determines validity of the real and fake images
        valid_adv = self.dis_siamese_net([source_emb_adv,target_emb_adv])

        self.adv_dis_siamese_model = Model(inputs=[source_img_adv, target_img_adv],
                            outputs=[valid_adv])
        self.adv_dis_siamese_model.compile(loss='binary_crossentropy',
                                        optimizer=optimizer)





        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the critic


        self.critic = self.build_discriminator()
        self.generator = self.build_generator()

        self.critic.trainable = True
        self.encoder.trainble = False
        self.disengtangled_encoder.trainble = False
        self.generator.trainable = False

        source_img = Input(shape=self.img_shape)
        target_img = Input(shape=self.img_shape)

        #Embedding of source and target
        # interpolate
        source_emb = self.encoder(source_img)
        target_emb = self.encoder(target_img)
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))

        ## disentangle feats
        disen_source_emb = self.disengtangled_encoder(source_img)
        disen_target_emb = self.disengtangled_encoder(target_img)

        disen_z_disc = Input(shape=(self.latent_dim,))

        gen_source_img = self.generator([source_emb,disen_source_emb])
        gen_target_img = self.generator([target_emb,disen_target_emb])
        gen_fake_img = self.generator([z_disc,disen_z_disc])

        # Discriminator determines validity of the real and fake images
        fake = self.critic(gen_fake_img)
        valid_source = self.critic(source_img)
        valid_target = self.critic(target_img)



        # Construct weighted average between real and fake emb
        interpolated_img_source = RandomWeightedAverage()([gen_fake_img, source_img])
        interpolated_img_target = RandomWeightedAverage()([gen_fake_img, target_img])
        # Determine validity of weighted sample
        validity_interpolated_source = self.critic(interpolated_img_source)
        validity_interpolated_target = self.critic(interpolated_img_target)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss_source = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img_source)
        partial_gp_loss_source.__name__ = 'gradient_penalty_source' # Keras requires function names
        partial_gp_loss_target = partial(self.gradient_penalty_loss,
                          averaged_samples=validity_interpolated_target)
        partial_gp_loss_target.__name__ = 'gradient_penalty_target' # Keras requires function names


        self.critic_model = Model(inputs=[source_img,target_img,z_disc,disen_z_disc],
                            outputs=[valid_source,valid_target, fake, validity_interpolated_source,validity_interpolated_target])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                               self.wasserstein_loss,
                                              partial_gp_loss_source,
                                              partial_gp_loss_target],
                                        optimizer=optimizer,
                                        loss_weights=[1 , 1 , 1 , 1 , 10])




        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        dis_z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator([z_gen,dis_z_gen])
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model([z_gen,dis_z_gen], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)



    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_encoder(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=(224,224,3), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(self.latent_dim))
        model.summary()
        img = Input(shape=self.img_shape)
        emb = model(img)
        return Model(img,emb)



    def build_siamese(self):

        model = Sequential()
        model.add(Dense(1024,input_shape=(self.latent_dim*2,)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(128))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(64))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1,activation='sigmoid'))

        inp1 = Input((self.latent_dim,))
        inp2 = Input((self.latent_dim,))
        inp = Concatenate()([inp1,inp2])
        out = model(inp)

        return Model([inp1,inp2],out)


    def repeat_block(self,input,n_time,scale,filters,kernel_size,strides,padding='same',activation='leaky_relu',kernel_regularizer=regularizers.l2(1e-4)):
        inp = input
        for i in range(n_time):
            if activation=='leaky_relu':
                net = Conv2D(filters,kernel_size, strides=strides, padding=padding,kernel_regularizer=kernel_regularizer)(inp)
                net = BatchNormalization()(net)
                net = LeakyReLU(alpha=0.1)(net)
            else:
                net = Conv2D(filters,kernel_size, strides=strides, padding=padding,kernel_regularizer=kernel_regularizer)(inp)
                net = BatchNormalization()(net)
                net = Activation(activation)(net)
            multi_scalar = Lambda(lambda x: x * scale)
            inp = Add()([multi_scalar(net),inp])
        return inp

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=(224,224,3), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1))
        model.summary()

        return model


    def build_generator(self,regularizers_variable=1e-4):

        latent = Input(shape=(self.latent_dim,))
        disentangle_latent = Input(shape=(self.latent_dim,))
        combine_latent = Concatenate()([latent,disentangle_latent])
        net = Dense(4096)(combine_latent)
        net = Reshape((4,4,256))(net)

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(128,kernel_size=[5, 5], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.3,128,kernel_size=[5, 5], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(64,kernel_size=[5, 5], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.3,64,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(32,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.3,32,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(16,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.3,16,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(8,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,3,0.1,8,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Cropping2D(cropping=((16, 16), (16, 16))) (net)
        net = Conv2D(3,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,3,0.1,3,kernel_size=[3, 3], strides=1, padding='same')
        net = Conv2D(3,kernel_size=[3, 3], strides=1,padding='same')(net)
        net = BatchNormalization()(net)
        net = Conv2D(3,kernel_size=[3, 3], strides=1,padding='same', activation = 'tanh')(net)
        return Model([latent,disentangle_latent], net)


    def read_image(self,path):
        return image.img_to_array(image.load_img(path, target_size=(self.img_rows,self.img_cols)))

    def get_data(self, paths):
        batch = len(paths)
        X = np.array([self.read_image(path) for path in paths])

        return X/255.0
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        X1 = self.get_data(self.paths[self.index[0]])
        X2 = self.get_data(self.paths[self.index[1]])

        ## same person
        X1_ = X1[1:,:,:,:]
        X12_ = X1[:-1,:,:,:]

        X2_ = X2[1:,:,:,:]
        X22_ = X2[:-1,:,:,:]

        ### concat
        Xsame = np.conatenate([X1_,X2_],axis=0)
        Xsame_  = np.conatenate([X12_,X22_],axis=0)
        ysame = np.ones(shape=(Xsame.shape[0]))

        ### diff person
        size = min(len(X1),len(X2))
        Xdiff = X1[:size,:,:,:]
        Xdiff_  = X2[:size,:,:,:]
        ydiff = np.zeros(shape=(size))


        return (Xsame,Xsame_,ysame),(Xdiff,Xdiff_,ydiff)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        dis_noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict([noise,dis_noise])

        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c,figsize=(20,20))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/mnt/hieuck/dataset/celebA/images/celebA_%d.png" % epoch)
        plt.close()


    def train(self, epochs, batch_size, sample_interval=50):




        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        # # Adversarial ground truths
        # valid = -np.ones((batch_size, 1))
        # fake =  np.ones((batch_size, 1))
        # dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty


        # # Evaluation ground truth
        # eval_valid = -np.ones((len(test_paths), 1))
        # eval_fake =  np.ones((len(test_paths), 1))
        # eval_dummy = np.zeros((len(test_paths), 1))
        # eval_imgs = self.get_data(test_paths)

        for epoch in range(epochs):
            indexes = random.sample(self.index,self.batch_size)
            while len(self.groups[indexes[0]]) == 1 or len(self.groups[indexes[1]]) == 1:
            	indexes = random.sample(self.index,self.batch_size)

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                (Xsame,Xsame_,ysame),(Xdiff,Xdiff_,ydiff) = self.__data_generation(indexes)
                # Train the encoder(siamese model)
                d_loss = self.siamese_model.train_on_batch([Xsame, Xsame_],
                                                                ysame)
                d_loss += self.siamese_model.train_on_batch([Xdiff, Xdiff_],
                                                                ydiff)
                # ---------------------
                #  Train Disentangle
                # ---------------------

                dis_loss = self.dis_siamese_model.train_on_batch([Xsame, Xsame_],
                                                                ysame)
                dis_loss += self.dis_siamese_model.train_on_batch([Xdiff, Xdiff_],
                                                                ydiff)
                # ---------------------
                #  Train Adv Disentangle
                # ---------------------

                adv_dis_loss = self.adv_dis_siamese_model.train_on_batch([Xsame, Xsame_],
                                                                ysame-1)
                adv_dis_loss += self.adv_dis_siamese_model.train_on_batch([Xdiff, Xdiff_],
                                                                ydiff+1)

                # ---------------------
                #  Train Critic
                # ---------------------

                valid = -np.ones((Xsame.shape[0], 1))
                fake =  np.ones((Xsame.shape[0], 1))
                dummy = np.zeros((Xsame.shape[0], 1)) # Dummy gt for gradient penalty

                valid_diff = -np.ones((Xdiff.shape[0], 1))
                fake_diff =  np.ones((Xdiff.shape[0], 1))
                dummy_diff = np.zeros((Xdiff.shape[0], 1)) # Dummy gt for gradient penalty


                z_disc = np.random.normal(0, 1, (Xsame.shape[0], self.latent_dim))
                disen_z_disc = np.random.normal(0, 1, (Xsame.shape[0], self.latent_dim))

                z_disc_diff = np.random.normal(0, 1, (Xdiff.shape[0], self.latent_dim))
                disen_z_disc_diff = np.random.normal(0, 1, (Xdiff.shape[0], self.latent_dim))

                critic_loss = self.critic_model.train_on_batch([Xsame, Xsame_,z_disc,disen_z_disc],
                                                                [valid,valid,fake,dummy,dummy])
                critic_loss += self.critic_model.train_on_batch([Xdiff, Xdiff_,z_disc_diff,disen_z_disc_diff],
                                                                [valid_diff,valid_diff,fake_diff,dummy_diff,dummy_diff])

                # ---------------------
                #  Train Generator
                # ---------------------

            g_loss = self.generator_model.train_on_batch([Xdiff,Xdiff_], valid_diff)

            g_loss += self.generator_model.train_on_batch([Xsame,Xsame_], valid)

            # Plot the progress
            print ("%d [En loss: %f] [dis En loss: %f] [dis adv En loss: %f] [Crit loss: %f] [G loss: %f]" % \
                          (epoch, d_loss,dis_loss,adv_dis_loss,critic_loss[0], g_loss))


            # ## Evaluate Process
            # eval_noise = np.random.normal(0, 1, (len(test_paths), self.latent_dim))

            # eval_d_loss = self.critic_model.evaluate([eval_imgs, eval_noise],
            #                                                     [eval_valid, eval_fake, eval_dummy])

            # eval_g_loss = self.generator_model.evaluate(eval_noise, eval_valid)
            # print ("%d [Eval D loss: %f] [Eval G loss: %f]" % (epoch, eval_d_loss[0], eval_g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                with open('/mnt/hieuck/dataset/celebA/model/wgan_gb.pkl', 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    wgan = Encoder_Disentagled()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)
