from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected,flatten,activation
from tflearn.layers.conv import conv_2d, max_pool_2d,conv_2d_transpose
from tflearn.layers.merge_ops import merge_outputs,merge
from tflearn.layers.estimator import regression
import numpy as np
import loadData

filedir = 'E:/AI/tensorflow/tensorflow/test/edgetest/HED-BSDS/train_pair_test.txt'
src_data_list, label_list = loadData.load_data(filedir)

data = loadData.read_images(src_data_list)  # image data
label = loadData.read_images(label_list)    # label

# Building 'VGG Network'
net_input_data = input_data(shape=[None,224,224,3])

conv1_1 = conv_2d(net_input_data, nb_filter = 64 , filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv1_2 = conv_2d(conv1_1,        nb_filter = 64 , filter_size = 3, strides=1, padding = 'same' ,activation='relu')
pool1 = max_pool_2d(conv1_2, kernel_size = 2, strides=2)

conv2_1 = conv_2d(pool1,          nb_filter = 128, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv2_2 = conv_2d(conv2_1,        nb_filter = 128, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
pool2 = max_pool_2d(conv2_2, kernel_size = 2, strides=2)

conv3_1 = conv_2d(pool2,          nb_filter = 256, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv3_2 = conv_2d(conv3_1,        nb_filter = 256, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv3_3 = conv_2d(conv3_2,        nb_filter = 256, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
pool3 = max_pool_2d(conv3_3, kernel_size = 2, strides=2)

conv4_1 = conv_2d(pool3,          nb_filter = 512, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv4_2 = conv_2d(conv4_1,        nb_filter = 512, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv4_3 = conv_2d(conv4_2,        nb_filter = 512, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
pool4 = max_pool_2d(conv4_3, kernel_size = 2, strides=2)

conv5_1 = conv_2d(pool4,          nb_filter = 512, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv5_2 = conv_2d(conv5_1,        nb_filter = 512, filter_size = 3, strides=1, padding = 'same' ,activation='relu')
conv5_3 = conv_2d(conv5_2,        nb_filter = 512, filter_size = 3, strides=1, padding = 'same' ,activation='relu')

# DSN conv 1
score_dsn1_up = conv_2d(conv1_2,  nb_filter = 1  , filter_size = 1, strides=1, padding = 'same' ,activation='relu')
dsn1_loss = regression(score_dsn1_up, optimizer='rmsprop',loss='categorical_crossentropy',learning_rate=0.0001)

# DSN conv 2
score_dsn2 = conv_2d(conv2_2,  nb_filter = 1  , filter_size = 1, strides = 1, padding = 'same' ,activation='relu')
score_dsn2_up = conv_2d_transpose(score_dsn2,  nb_filter = 1  , filter_size = 4,output_shape=[224,224], strides = 2, padding = 'same' ,activation='relu')
dsn2_loss = regression(score_dsn2_up, optimizer='rmsprop',loss='categorical_crossentropy',learning_rate=0.0001)

# DSN conv 3
score_dsn3 = conv_2d(conv3_3,  nb_filter = 1  , filter_size = 1, strides = 1, padding = 'same' ,activation='relu')
score_dsn3_up = conv_2d_transpose(score_dsn3,  nb_filter = 1  , filter_size = 8, output_shape=[224,224],strides = 4, padding = 'same' ,activation='relu')
dsn3_loss = regression(score_dsn3_up, optimizer='rmsprop',loss='categorical_crossentropy',learning_rate=0.0001)

# DSN conv 4
score_dsn4 = conv_2d(conv4_3,  nb_filter = 1  , filter_size = 1, strides = 1, padding = 'same' ,activation='relu')
score_dsn4_up = conv_2d_transpose(score_dsn4,  nb_filter = 1  , filter_size = 16, output_shape=[224,224],strides = 8, padding = 'same' ,activation='relu')
dsn4_loss = regression(score_dsn4_up, optimizer='rmsprop',loss='categorical_crossentropy',learning_rate=0.0001)

# DSN conv 5
score_dsn5 = conv_2d(conv5_3,  nb_filter = 1  , filter_size = 1, strides = 1, padding = 'same' ,activation='relu')
score_dsn5_up = conv_2d_transpose(score_dsn5,  nb_filter = 1  , filter_size = 32, output_shape=[224,224],strides = 16, padding = 'same' ,activation='relu')
dsn5_loss = regression(score_dsn5_up, optimizer='rmsprop',loss='categorical_crossentropy',learning_rate=0.0001)

concat_upscore = merge([score_dsn1_up,score_dsn2_up,score_dsn3_up,score_dsn4_up,score_dsn5_up],mode='concat',axis =3)#channel
upscore_fuse = conv_2d(concat_upscore,  nb_filter = 3  , filter_size = 1, strides = 1, padding = 'same' ,activation='relu')
fuse_loss = regression(upscore_fuse, optimizer='rmsprop',loss='categorical_crossentropy',learning_rate=0.0001)


model = tflearn.DNN(fuse_loss,checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)

model.fit(data, label, n_epoch=10, shuffle=True,
          show_metric=True, batch_size=1, snapshot_step=10,
          snapshot_epoch=False, run_id='vgg_edge_dect')
model.save('edge_model.tflearn')
