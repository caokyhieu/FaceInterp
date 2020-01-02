from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model,load_model,Sequential
from keras.layers import Input,Dense,Conv2D,BatchNormalization,Add,LeakyReLU,Flatten,Lambda,Reshape,UpSampling2D,Cropping2D,Subtract, ReLU,Dropout
from keras.layers import ZeroPadding2D,Concatenate,Activation,Concatenate,GlobalAveragePooling2D
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
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import to_categorical

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"

K.set_image_data_format('channels_last')

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        shape = K.shape(inputs[0])
        alpha = K.random_uniform((shape[0], 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class Interpolate(_Merge):
    """Provides a interpolate for images and emb features"""
    def _merge_function(self, inputs):
        alpha = inputs[2]
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class Encoder_Disentagled():
    def __init__(self):
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        ## Init params
        session = K.get_session()
        init = tf.global_variables_initializer()
        session.run(init)


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


        # Build the generator and critic
        self.shared_net = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.img_shape, pooling=None)

        self.encoder = self.build_encoder(self.shared_net)
        self.disengtangled_encoder = self.build_encoder(self.shared_net)
        self.siamese_net = self.build_classifier()

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

        # Discriminator determines validity of the real and fake images
        valid_source = self.siamese_net(source_emb)
        valid_target = self.siamese_net(target_emb)

        self.siamese_model = Model(inputs=[source_img, target_img],
                            outputs=[valid_source,valid_target])
        self.siamese_model.compile(loss=['categorical_crossentropy','categorical_crossentropy'],
                                        optimizer=optimizer)

        #-------------------------------
        # Construct Computational Graph
        #         for Disentangled feats
        #-------------------------------

        self.dis_siamese_net = self.build_classifier()
        self.disengtangled_encoder.trainable= False

        dis_source_img = Input(shape=self.img_shape)
        dis_target_img = Input(shape=self.img_shape)

        #Embedding of source and target
        dis_source_emb = self.disengtangled_encoder(dis_source_img)
        dis_target_emb= self.disengtangled_encoder(dis_target_img)

        # # Construct weighted average between real and fake emb
        # interpolated_emb_adv = RandomWeightedAverage()([source_emb_adv, target_emb_adv])


        # Discriminator determines validity of the real and fake images
        dis_valid_source = self.dis_siamese_net(dis_source_emb)
        dis_valid_target = self.dis_siamese_net(dis_target_emb)



        self.dis_siamese_model = Model(inputs=[dis_source_img, dis_target_img],
                            outputs=[dis_valid_source,dis_valid_target])
        self.dis_siamese_model.compile(loss=['categorical_crossentropy','categorical_crossentropy'],
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
        valid_adv_source = self.dis_siamese_net(source_emb_adv)
        valid_adv_target = self.dis_siamese_net(target_emb_adv)

        self.adv_dis_siamese_model = Model(inputs=[source_img_adv, target_img_adv],
                            outputs=[valid_adv_source,valid_adv_target])
        self.adv_dis_siamese_model.compile(loss=['categorical_crossentropy','categorical_crossentropy'],
                                        optimizer=optimizer)


        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------


        # For the generator we freeze the critic's layers
        self.critic = self.build_discriminator()
        self.generator = self.build_generator()
        self.encoder.trainble = False
        self.disengtangled_encoder.trainble = False

        self.critic.trainable = False
        self.generator.trainable = True


        source_img = Input(shape=self.img_shape)
        target_img = Input(shape=self.img_shape)

        #Embedding of source and target
        # interpolate
        source_emb = self.encoder(source_img)
        target_emb = self.encoder(target_img)

        ## disentangle feats
        disen_source_emb = self.disengtangled_encoder(source_img)
        disen_target_emb = self.disengtangled_encoder(target_img)

        alpha1 = Input(shape=(1,))
        alpha2 = Input(shape=(1,))
        interp_layer_emb = Interpolate()([source_emb,target_emb,alpha1])
        interp_layer_disen_emb = Interpolate()([disen_source_emb,disen_target_emb,alpha2])

        #Generate images based of iterpolate
        # img = self.generator([z_gen,dis_z_gen])
        img = self.generator([interp_layer_emb,interp_layer_disen_emb])
        # Discriminator determines validity
        valid = self.critic(img)
        # valid_prime = self.critic(img_prime)
        # Defines generator model
        self.generator_model_1 = Model([source_img,target_img,alpha1,alpha2], [img,valid])

        ## add the L1 reconstruction Loss

        ## input
        source_img = Input(shape=self.img_shape)
        target_img = Input(shape=self.img_shape)
        alpha1 = Input(shape=(1,))
        alpha2 = Input(shape=(1,))

        inverse = Lambda(lambda x: 1/x)
        inverse_alpha1 = inverse(alpha1)
        inverse_alpha2 = inverse(alpha2)
        img,valid = self.generator_model_1([source_img,target_img,alpha1,alpha2])
        img_prime,_ = self.generator_model_1([img,target_img,inverse_alpha1,inverse_alpha2])

        self.generator_model = Model([source_img,target_img,alpha1,alpha2],[img_prime,valid])
        self.generator_model.compile(loss=[self.L1_loss,self.wasserstein_loss], optimizer=optimizer)


        # For the critic

        self.critic.trainable = True
        self.generator_model_1.trainable = False

        source_img = Input(shape=self.img_shape)
        target_img = Input(shape=self.img_shape)
        alpha1 = Input(shape=(1,))
        alpha2 = Input(shape=(1,))

        # interpolate image
        gen_fake_img,_ = self.generator_model_1([source_img,target_img,alpha1,alpha2])


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


        self.critic_model = Model(inputs=[source_img,target_img,alpha1,alpha2],
                            outputs=[valid_source,valid_target, fake, validity_interpolated_source,validity_interpolated_target])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                               self.wasserstein_loss,
                                              partial_gp_loss_source,
                                              partial_gp_loss_target],
                                        optimizer=optimizer,
                                        loss_weights=[1 , 1 , 3 , 5 , 5])



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


    def build_encoder(self,shared_net):

        # model = Sequential()
        #
        # model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(1024, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Flatten())
        # model.add(Dense(self.latent_dim,activation='sigmoid'))
        # model.summary()
        # img = Input(shape=self.img_shape)
        # emb = model(img)
        # model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.img_shape, pooling=None)

        out = shared_net.output
        out = self.subnet(out)

        return Model(shared_net.input,out)


    def subnet(self,input):
        net = Conv2D(1024, kernel_size=3, strides=1, padding="same") (input)
        net = Conv2D(512, kernel_size=3, strides=1, padding="same") (net)
        net = Conv2D(512, kernel_size=3, strides=1, padding="same") (net)
        net = GlobalAveragePooling2D()(net)
        net =Dense(self.latent_dim)(net)
        return net


    def build_classifier(self):
        inp = Input((self.latent_dim,))
        out = Dense(len(self.index),activation='softmax')(inp)
        return Model(inp,out)


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

    def L1_loss(self,y_true,y_pred):
        x = K.abs(y_true - y_pred)
        return K.mean(x,axis=(0,1,2,3)) * 100

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
        net = self.repeat_block(net,6,0.7,128,kernel_size=[5, 5], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(64,kernel_size=[5, 5], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.7,64,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(32,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.7,32,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(16,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.7,16,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Conv2D(8,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,6,0.7,8,kernel_size=[3, 3], strides=1, padding='same')

        net = UpSampling2D(size=(2,2)) (net)
        net = Cropping2D(cropping=((16, 16), (16, 16))) (net)
        net = Conv2D(3,kernel_size=[3, 3], strides=1, padding='same',kernel_regularizer=regularizers.l2(regularizers_variable))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = self.repeat_block(net,3,0.7,3,kernel_size=[3, 3], strides=1, padding='same')

        net = Conv2D(3,kernel_size=[3, 3], strides=1,padding='same')(net)
        net = BatchNormalization()(net)
        net = Conv2D(3,kernel_size=[3, 3], strides=1,padding='same',activation='tanh')(net)
        model = Model([latent,disentangle_latent], net)
        model.summary()
        return model


    def read_image(self,path):
        return image.img_to_array(image.load_img(path, target_size=(self.img_rows,self.img_cols)))

    def get_data(self, paths):
        batch = len(paths)
        X = np.array([self.read_image(path) for path in paths])

        return X/127.5 -1

    def __data_generation(self):
        half_batch = self.batch_size//2
        index_X1 = random.sample(self.paths.index.tolist(),half_batch)
        index_X2 = random.sample(self.paths.index.tolist(),half_batch)
        X1 = self.get_data(self.paths.loc[index_X1][2].values)
        y1 = self.paths.loc[index_X1][1].values - 1
        X2 = self.get_data(self.paths.loc[index_X2][2].values)
        y2 = self.paths.loc[index_X2][1].values - 1
        return X1,y1,X2,y2

    def sample_images(self, epoch,X1,X2,alpha):

        gen_imgs,_ = self.generator_model_1.predict([X1,X2,alpha[:,0],alpha[:,1]])
        # gen_imgs_diff,_ = self.generator_model_1.predict([Xdiff,Xdiff_,alpha_diff[:,0],alpha_diff[:,1]])
        gen_imgs = (gen_imgs+1)/2.0
        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        # gen_imgs = np.concatenate((gen_imgs_same,gen_imgs_diff),axis=0)
        total = gen_imgs.shape[0]
        c = 5
        r = total//c + 1
        fig, axs = plt.subplots(r, c,figsize=(20,20))
        cnt = 0
        for i in range(r):
            for j in range(0,min(c,total - i * c)):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/mnt/hieuck/dataset/celebA/images/celebA_%d.png" % epoch)
        plt.close()


    def train(self, epochs, sample_interval=50):

        half_batch = self.batch_size//2

        for epoch in range(epochs):
            # indexes = random.sample(self.index,self.batch_size)
            # while len(self.groups[indexes[0]]) < 3 or len(self.groups[indexes[1]]) < 3:
            # 	indexes = random.sample(self.index,self.batch_size)


            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                X1,y1,X2,y2 = self.__data_generation()
                # Train the encoder(siamese model)
                label_y1 = to_categorical(y1, num_classes=len(self.index), dtype='float32')
                label_y2 = to_categorical(y2, num_classes=len(self.index), dtype='float32')
                d_loss = self.siamese_model.train_on_batch([X1, X2],[label_y1,label_y2])
                # d_loss += self.siamese_model.train_on_batch([Xdiff, Xdiff_],[ydiff])
                # ---------------------
                #  Train Disentangle
                # ---------------------

                dis_loss = self.dis_siamese_model.train_on_batch([X1, X2],[label_y1,label_y2])
                # dis_loss += self.dis_siamese_model.train_on_batch([Xdiff, Xdiff_],[ydiff])
                # ---------------------
                #  Train Adv Disentangle
                # ---------------------

                adv_dis_loss = self.adv_dis_siamese_model.train_on_batch([X1, X2],[np.tile([1/len(self.index)],(half_batch,len(self.index))),\
                                                                            np.tile([1/len(self.index)],(half_batch,len(self.index)))])

                # adv_dis_loss += self.adv_dis_siamese_model.train_on_batch([Xdiff, Xdiff_],[ydiff+0.5])

                # ---------------------
                #  Train Critic
                # ---------------------

                valid = -np.ones((half_batch, 1))
                fake =  np.ones((half_batch, 1))
                dummy = np.zeros((half_batch, 1)) # Dummy gt for gradient penalty
                alpha = np.random.uniform(0.3,0.9,(half_batch,2))
                # valid_diff = -np.ones((Xdiff.shape[0], 1))
                # fake_diff =  np.ones((Xdiff.shape[0], 1))
                # dummy_diff = np.zeros((Xdiff.shape[0], 1)) # Dummy gt for gradient penalty
                # alpha_same = np.random.uniform(0.3,0.9,(Xsame.shape[0],2))
                # alpha_diff = np.random.uniform(0.3,0.9,(Xdiff.shape[0],2))


                critic_loss = self.critic_model.train_on_batch([X1, X2,alpha[:,0],alpha[:,1]],
                                                                [valid,valid,fake,dummy,dummy])
                # critic_loss += self.critic_model.train_on_batch([Xdiff, Xdiff_,alpha_diff[:,0],alpha_diff[:,1]],
                #                                                 [valid_diff,valid_diff,fake_diff,dummy_diff,dummy_diff])

                # ---------------------
                #  Train Generator
                # ---------------------

            g_loss = self.generator_model.train_on_batch([X1,X2,alpha[:,0],alpha[:,1]], [X1,valid])

            # g_loss += self.generator_model.train_on_batch([Xdiff,Xdiff_,alpha_diff[:,0],alpha_diff[:,1]], [Xdiff,valid_diff])

            # Plot the progress
            print ("%d [En loss: %f] [dis En loss: %f] [dis adv En loss: %f] [Crit loss: %f] [Rescon_L1: %f] [G loss: %f]" % \
                          (epoch, d_loss[0],dis_loss[0],adv_dis_loss[0],critic_loss[3], g_loss[1], g_loss[2]))


            # ## Evaluate Process
            # eval_noise = np.random.normal(0, 1, (len(test_paths), self.latent_dim))

            # eval_d_loss = self.critic_model.evaluate([eval_imgs, eval_noise],
            #                                                     [eval_valid, eval_fake, eval_dummy])

            # eval_g_loss = self.generator_model.evaluate(eval_noise, eval_valid)
            # print ("%d [Eval D loss: %f] [Eval G loss: %f]" % (epoch, eval_d_loss[0], eval_g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch,X1,X2,alpha)
                with open('/mnt/hieuck/dataset/celebA/model/wgan_gb.pkl', 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    wgan = Encoder_Disentagled()
    wgan.train(epochs=1000000, sample_interval=100)
