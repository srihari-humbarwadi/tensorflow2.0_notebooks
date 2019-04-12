#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
print('Tensorflow version =', tf.__version__)


# ## Creating data generators
#  - Random rotations applied to augment data

# In[10]:


train_data_dir = 'dataset/Training'
validation_data_dir = 'dataset/Validation'
batch_size = 32
img_height, img_width = 100, 100

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=30)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='binary')

class_map = train_generator.class_indices


# ## Visualizing some images
#  - The dataset containes 50000 benign images and 50000 positive cancer images
#  - Our task here is to classify a image as 'benign' / 'cancer' given a image

# In[3]:


plt.subplots(2, 4, figsize=(8, 5))
disp_images, labels = train_generator.__getitem__(1)
for i, idx in enumerate(disp_images[:8]):
    plt.subplot(2, 4, i+1)
    plt.title(f'Class: {labels[i]}')
    plt.axis('off')
    plt.imshow(disp_images[i])


# ## Building the model
#  - We select the pretrained InceptionV3 model available in keras applications package
#  - The last layers is removed and a new fully connected layers is added with units=num_classes (here 2)
# 

# In[4]:


base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, pooling='avg')
x = base_model.output
predictions = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, predictions)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])


# ## Creating call back objects to help us visualize training and do some trivial tasks like checkpointing, early stopping and training

# In[6]:


tensorboard = TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='batch')
callbacks_list = [ModelCheckpoint('model_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True),
                  tensorboard, EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)]


# ## Training the model

# In[7]:


model.fit_generator(train_generator, 
                    epochs=10, 
                    validation_data=validation_generator, 
                    callbacks=callbacks_list, 
                    workers=8, 
                    max_queue_size=300)


# ## Loading the best weights to perform inference

# In[8]:


model.load_weights('model_weights.h5')


# # Inference loop
# 

# In[18]:


plt.subplots(2, 5, figsize=(16, 10))
test_images, labels = validation_generator.__getitem__(1)
for i, idx in enumerate(test_images[:10]):
    img = test_images[i]
    pred_prob = np.squeeze(model.predict(np.expand_dims(img, axis=0)))
    pred = class_map['1'] if pred_prob >= 0.5 else class_map['0'] 
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.title(f'True label: {labels[i]}\nPredicted label: {pred}', wrap=True)
    plt.imshow(test_images[i])

