#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import keras
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
import cifair
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D,Reshape
from keras.layers import Concatenate,MaxPooling1D
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import backend as k
from keras.constraints import Constraint


# In[2]:


num_classes = 10
rows, cols = 32, 32
channels = 3


# In[3]:


(x_train, y_train), (x_test, y_test) = cifair.load_cifair10()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[4]:


def preprocess_data(data_set):
    mean = np.mean(x_train,axis=0)
    std = np.std(x_train,axis=0)

    data_set -= mean
    data_set /= std
    return data_set


# In[5]:


x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)


# In[6]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.10)


# In[7]:


datagen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.125,
    height_shift_range=0.125,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.10
)
datagen_train.fit(x_train)
num_filter = 40 
compression = 1.0
dropout_rate = 0.20
l = 6


# In[8]:


class WeightClip(Constraint):
    def __init__(self, c=100):
        self.c = c
    def __call__(self, p):    
        return K.clip(p, 0, self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,'c': self.c}

def pos_reg(weight_matrix):
        return 0.5 * K.max((K.zeros(K.shape(weight_matrix)),-weight_matrix))


# In[13]:


def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('elu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=True ,padding='same',W_constraint = WeightClip(2))(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp

def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('elu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=True ,padding='same',W_constraint = WeightClip(2))(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg

def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('elu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax',W_constraint = WeightClip(2))(flat)
    return output


def add_denseblock_NN(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp

def add_transition_NN(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    dropout_rate = 0.2
    BatchNorm = BatchNormalization()(input)
    relu = Activation('elu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg

def output_layer_NN(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    return output


input = Input(shape=(img_height, img_width, channel,))
select_model = "IOC"
if (select_model == 'IOC'):                               
    First_Conv2D = Conv2D(num_filter*2, (3,3), use_bias=False ,padding='same')(input)

    First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
    First_Transition = add_transition(First_Block, num_filter, dropout_rate)

    Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
    Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

    Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
    Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

    Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
    output = output_layer(Last_Block)
else:                               
    First_Conv2D = Conv2D(num_filter*2, (3,3), use_bias=False ,padding='same')(input)

    First_Block = add_denseblock_NN(First_Conv2D, num_filter, dropout_rate)
    First_Transition = add_transition_NN(First_Block, num_filter, dropout_rate)

    Second_Block = add_denseblock_NN(First_Transition, num_filter, dropout_rate)
    Second_Transition = add_transition_NN(Second_Block, num_filter, dropout_rate)

    Third_Block = add_denseblock_NN(Second_Transition, num_filter, dropout_rate)
    Third_Transition = add_transition_NN(Third_Block, num_filter, dropout_rate)

    Last_Block = add_denseblock_NN(Third_Transition,  num_filter, dropout_rate)
    output = output_layer_NN(Last_Block)

model = Model(inputs=[input], outputs=[output])
model.summary()


# In[15]:


reduce_lr = ReduceLROnPlateau(monitor = 'validation_loss', factor = 0.1, patience = 5, min_lr = 0.000001)
early_stop = EarlyStopping(monitor = "validation_loss", patience = 10)


# In[15]:


def decay_fn(epoch, lr):
    if epoch < 250:
        return 0.001
    elif epoch >= 250 and epoch < 500:
        return 0.0001
    else:
        return 0.00001


# In[16]:


lr_scheduler = LearningRateScheduler(decay_fn)
csv_logger = CSVLogger('mixture_training_'+select_model+'.log')
filepath = "{epoch:03d}-{val_accuracy:.3f}_"+select_model+".hdf5"
model_chkpt = ModelCheckpoint(filepath, monitor = "val_accuracy", save_best_only=True, verbose = 0)


# In[ ]:


history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=(len(x_train)/batch_size)*5, epochs=epochs, verbose = 0, validation_data=(x_val, y_val) , callbacks = [lr_scheduler, csv_logger, model_chkpt])


# In[ ]:


train_log = pd.read_csv("./mixture_training_"+select_model+".log")
train_log.head(100)

plt.ioff()
#plt.subplot(1, 2, 1)
plt.figure()
plt.plot(train_log['accuracy'])
plt.plot(train_log['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./acc.png')

#plt.subplot(1, 2, 2)
plt.figure()
plt.plot(train_log['loss'])
plt.plot(train_log['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./loss.png')
plt.close(fig)

