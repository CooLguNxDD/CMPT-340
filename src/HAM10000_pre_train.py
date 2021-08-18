from helper import plot_model_accuracy, plot_model_accuracy

import os
import gc
from glob import glob
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import efficientnet.tfkeras as ef
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization


# Load data
df = pd.read_csv('../data/HAM10000_metadata.csv', delimiter=',')
df.dataframeName = 'HAM10000_metadata.csv'


# Data preprocessing
label_encoder = LabelEncoder()
label_encoder.fit(df['dx'])
print(list(label_encoder.classes_))
df['label'] = label_encoder.transform(df["dx"])
print(df.sample(5))

features_dict = {0:"akiex",1:"bcc",2:"bkl",3:"df", 4:"mel",5:"nv",6:"vasc"}

image_size = 150 #the size that the image will resize to
image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('../data/','*','*.jpg'))}
#image path
df['path'] = df['image_id'].map(lambda id: image_path.get(id))
#it takes time to process the image
df['image_64'] = df['path'].map(lambda path:Image.open(path).resize((image_size,image_size)))

df['image'] = df['image_64'].map(lambda image: np.asarray(image))
data = np.asarray(df['image'].to_list()).astype("short")
label_to_one_hot = to_categorical(df['label'], num_classes=7)

# Split testign and training data
#80% data for training, 20% data for testing
train_data,test_data,train_label,test_label = train_test_split(data,label_to_one_hot,test_size=0.2,random_state=87,stratify=label_to_one_hot)
#80% train data for training, 20% train data for validation
train_data,valid_data,train_label,valid_label = train_test_split(train_data,train_label,test_size=0.2,random_state=87,stratify=train_label)

# Center and normalize the images
data_gen = ImageDataGenerator(rescale=1.0/255., featurewise_center=True,
    rotation_range=50 ,horizontal_flip=True,vertical_flip=True,
    height_shift_range=0.2,width_shift_range=0.2, shear_range=0.2)
data_gen.fit(train_data)

valid_data_gen = ImageDataGenerator(rescale=1.0/255., featurewise_center=True,
    rotation_range=50 ,horizontal_flip=True,vertical_flip=True,
    height_shift_range=0.2,width_shift_range=0.2, shear_range=0.2)
valid_data_gen.fit(valid_data)

test_data = test_data/255.0


# Model building
# reference: https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
num_classes = 7

vgg16_model = VGG16(input_shape=(image_size,image_size,3), include_top = False, weights='imagenet')

# don't use the pre-trained layers
for layer in vgg16_model.layers:
    layer.trainable = False

#Flatten layer
temp_layer = BatchNormalization()(vgg16_model.output)
temp_layer = Flatten()(temp_layer)

#Dense layer 1
temp_layer = Dense(2048,activation='relu')(temp_layer)

#Dense layer 2
temp_layer = Dense(1024,activation='relu')(temp_layer)

#output layer
temp_layer = Dense(7,activation='softmax')(temp_layer)

model = Model(vgg16_model.input,temp_layer)
model.summary()

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

check_point = ModelCheckpoint("../models/VGG_16.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto')

try:
    model.load_weights('../models/VGG_16_weight.h5')
except:
    print("new model")
epochs = 200

train_history = model.fit(
    data_gen.flow(train_data,train_label,batch_size=128),
    validation_data=valid_data_gen.flow(valid_data,valid_label,batch_size=32),
    epochs=epochs,
    steps_per_epoch=len(train_data)/128,
    verbose=2,callbacks=[check_point,early_stop])

model.save_weights('../models/VGG_16_weight.h5')


# Model evaluation
plot_model_accuracy(train_history,'../plots/VGG_16_training_history.jpg')
plot_model_loss(train_history, 'loss', '../plots/VGG_16_training_loss.jpg')
score = model.evaluate(test_data, test_label)
print('Test accuracy:', score[1])
prediction = model.predict(test_data)
prediction_class = np.argmax(prediction,axis=1)
prediction_label = np.argmax(test_label,axis=1)
mapping = lambda x:features_dict[x]
pred_class_to_feature = np.array([mapping(x) for x in prediction_class])
pred_label_to_feature = np.array([mapping(x) for x in prediction_label])
print(pd.crosstab(pred_label_to_feature,pred_class_to_feature,rownames=['actual'],colnames=['predicted']))

# Model Building (InceptionV3 Model)
# reference: https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
num_classes = 7
dimension = image_size*image_size*3

InceptionV3_model = InceptionV3(input_shape=(image_size,image_size,3), include_top=False, weights='imagenet')

# don't use the pre-trained model
for layer in InceptionV3_model.layers:
    layer.trainable = False

#Flatten layer
temp_layer = BatchNormalization()(InceptionV3_model.output)
temp_layer = Flatten()(temp_layer)

#Dense layer 1
temp_layer = Dense(1024,activation='relu')(temp_layer)

#output layer
temp_layer = Dense(7,activation='softmax')(temp_layer)

model = Model(InceptionV3_model.input,temp_layer)
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

check_point = ModelCheckpoint("../models/InceptionV3.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto')

try:
    model.load_weights('../models/InceptionV3_weight.h5')
except:
    print("new model")

epochs = 50

train_history = model.fit(
    data_gen.flow(train_data,train_label,batch_size=128),
    validation_data=valid_data_gen.flow(valid_data,valid_label,batch_size=32),
    epochs=epochs,
    steps_per_epoch=len(train_data)/128,
    verbose=2,callbacks=[check_point,early_stop])

model.save_weights('../models/InceptionV3_weight.h5')


# Model Evaluation
plot_model_accuracy(train_history,'../plots/InceptionV3_training_history.jpg')
plot_model_loss(train_history, 'loss', '../plots/InceptionV3_training_loss.jpg')
score = model.evaluate(test_data, test_label)
print('Test accuracy:', score[1])
prediction = model.predict(test_data)
prediction_class = np.argmax(prediction,axis=1)
prediction_label = np.argmax(test_label,axis=1)
mapping = lambda x:features_dict[x]
pred_class_to_feature = np.array([mapping(x) for x in prediction_class])
pred_label_to_feature = np.array([mapping(x) for x in prediction_label])
print(pd.crosstab(pred_label_to_feature,pred_class_to_feature,rownames=['actual'],colnames=['predicted']))


# Model Building (Efficient Net Model)
# reference: https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
num_classes = 7
dimension = image_size*image_size*3

EN_model = ef.EfficientNetB0(input_shape=(image_size,image_size,3), include_top=False, weights='imagenet')
for layer in EN_model.layers:
    layer.trainable = False

#Flatten layer
temp_layer = BatchNormalization()(EN_model.output)
temp_layer = Flatten()(temp_layer)

#Dense layer 1
temp_layer = Dense(1024,activation='relu')(temp_layer)

#output layer
temp_layer = Dense(7,activation='softmax')(temp_layer)

model = Model(EN_model.input,temp_layer)
#model.summary()

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

check_point = ModelCheckpoint("../models/Efficient_Net.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto')

try:
    model.load_weights('../models/Efficient_Net_weight.h5')
except:
    print("new model")
epochs = 100

train_history = model.fit(
    data_gen.flow(train_data,train_label,batch_size=128),
    validation_data=valid_data_gen.flow(valid_data,valid_label,batch_size=32),
    epochs=epochs,
    steps_per_epoch=len(train_data)/128,
    verbose=2,callbacks=[check_point,early_stop])

model.save_weights('../models/Efficient_Net_weight.h5')


# Model evaluation
plot_model_accuracy(train_history,'../plots/Efficient_Net_training_history.jpg')
plot_model_loss(train_history, 'loss', '../plots/Efficient_Net_training_loss.jpg')
score = model.evaluate(test_data, test_label)
print('Test accuracy:', score[1])
prediction = model.predict(test_data)

prediction_class = np.argmax(prediction,axis=1)
prediction_label = np.argmax(test_label,axis=1)

mapping = lambda x:features_dict[x]
pred_class_to_feature = np.array([mapping(x) for x in prediction_class])
pred_label_to_feature = np.array([mapping(x) for x in prediction_label])

print(pd.crosstab(pred_label_to_feature,pred_class_to_feature,rownames=['actual'],colnames=['predicted']))
