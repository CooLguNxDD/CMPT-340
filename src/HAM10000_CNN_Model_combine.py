from helper import plot_model_accuracy, plot_model_accuracy

import os
import math
import random
from glob import glob
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split


# Load data
df = pd.read_csv('../data/HAM10000_metadata.csv', delimiter=',')
df.dataframeName = 'HAM10000_metadata.csv'

# Preprocess data
label_encoder = LabelEncoder()
label_encoder.fit(df['dx'])
print(list(label_encoder.classes_))
df['label'] = label_encoder.transform(df["dx"])
print(df.sample(5))

features_dict = {0:"akiex",1:"bcc",2:"bkl",3:"df",4:"mel",5:"nv",6:"vasc"}

total_data = 0
num_classes = 7
for num_of_data in df['label'].value_counts():
    total_data+=num_of_data
print("total number of data: ",total_data)
print(df['label'].value_counts())
#calc weight for each class
class_weight ={}
for index, value in df.label.value_counts().iteritems():
    v1 = math.log(total_data/(float(value)*num_classes))
    class_weight[index] = v1 if v1>1.0 else 1.0

for i in class_weight:
    print(i," ",class_weight[i])

image_size = 32 #the size that the image will resize to
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('../data/','*','*.jpg'))}
#image path
df['path'] = df['image_id'].map(lambda id: image_path.get(id))

#image_64 is the actual image(resized to 64x64) store in the dataframe
#it takes time to process the image
df['image_64'] = df['path'].map(lambda path:Image.open(path).resize((image_size,image_size)))

#print some image
index = 1
for image in df['image_64'].head(10):
    plots = plt.subplot(2,5,index)
    plots.imshow(image)
    index+=1

df['image'] = df['image_64'].map(lambda image: np.asarray(image))
data = np.asarray(df['image'].to_list())
#data = data.reshape(total_data,image_size*image_size*3).astype('float32')
data = data/255.0 #normalise the RGB value to [0...1]
label_to_one_hot = to_categorical(df['label'], num_classes=7)
#80% data for training
#20% data for testing
train_data,test_data,train_label,test_label = train_test_split(data,label_to_one_hot,test_size=0.20,random_state=87,stratify=label_to_one_hot)

num_classes = 7
dimension = image_size*image_size*3


# Model building (versio 1)
model_CNN_V1 = Sequential()

#CNN layer 1:
model_CNN_V1.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(image_size,image_size,3),activation='relu',padding='same'))
model_CNN_V1.add(Dropout(0.1))
model_CNN_V1.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V1.add(MaxPooling2D(pool_size=(2,2)))

#CNN layer 2
model_CNN_V1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V1.add(Dropout(0.1))
model_CNN_V1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V1.add(MaxPooling2D(pool_size=(2,2)))

#CNN layer 3
model_CNN_V1.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V1.add(Dropout(0.1))
model_CNN_V1.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V1.add(MaxPooling2D(pool_size=(2,2)))

#Flatten layer
model_CNN_V1.add(Flatten())

#hidden layer 1
model_CNN_V1.add(Dense(2048,activation='relu'))
model_CNN_V1.add(Dropout(0.2))

#hidden layer 2
model_CNN_V1.add(Dense(1024,activation='relu'))
model_CNN_V1.add(Dropout(0.2))

#output layer
model_CNN_V1.add(Dense(7,activation='softmax'))
model_CNN_V1.summary()

model_CNN_V1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

try:
    model_CNN_V1.load_weights('../models/CNN_weight.h5')
except:
    print("new model_CNN_V1")

batch_size = 256
epochs = 200

train_history = model_CNN_V1.fit(
    train_data,train_label,
    epochs=epochs,
    validation_split=0.2,
    batch_size=batch_size,
    verbose=2,class_weight=class_weight)

model_CNN_V1.save_weights('../models/CNN_weight.h5')
model_CNN_V1.save('../models/CNN_V1.h5')

# apply oversampling
smote_sample = SMOTE(random_state=87)
train_data = train_data.reshape(-1,image_size*image_size*3)
train_data_oversample, train_label_oversample = smote_sample.fit_resample(train_data, train_label)
train_data_oversample = train_data_oversample.reshape(-1,image_size,image_size,3)
train_data = train_data.reshape(-1,image_size,image_size,3)


# Model building (versio 2)
num_classes = 7
dimension = image_size*image_size*3

model_CNN_V2 = Sequential()

#CNN layer 1:
model_CNN_V2.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(image_size,image_size,3),activation='relu',padding='same'))
model_CNN_V2.add(Dropout(0.2))
model_CNN_V2.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V2.add(MaxPooling2D(pool_size=(2,2)))

#CNN layer 2
model_CNN_V2.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V2.add(Dropout(0.2))
model_CNN_V2.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V2.add(MaxPooling2D(pool_size=(2,2)))

#CNN layer 3
model_CNN_V2.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V2.add(Dropout(0.2))
model_CNN_V2.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model_CNN_V2.add(MaxPooling2D(pool_size=(2,2)))

#Flatten layer
model_CNN_V2.add(Flatten())
model_CNN_V2.add(Dropout(0.4))

#hidden layer 1
model_CNN_V2.add(Dense(2048,activation='relu'))
model_CNN_V2.add(Dropout(0.4))

#hidden layer 2
model_CNN_V2.add(Dense(1024,activation='relu'))
model_CNN_V2.add(Dropout(0.4))

#output layer
model_CNN_V2.add(Dense(7,activation='softmax'))
model_CNN_V2.summary()

model_CNN_V2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

try:
    model_CNN_V2.load_weights('../models/CNN_V2_weight.h5')
except:
    print("new model_CNN_V2")

# Train
batch_size = 256
epochs = 200

train_history_2 = model_CNN_V2.fit(
    train_data_oversample,train_label_oversample,
    epochs=epochs,
    batch_size= batch_size,
    validation_split=0.2,
    verbose=2,shuffle=True)

model_CNN_V2.save_weights('../models/CNN_V2_weight.h5')
model_CNN_V2.save('../models/CNN_V2.h5')

# Model evaludation
#plot train history
print("V1")
plot_model_accuracy(train_history,'../plots/CNN_V1_training_history.jpg')
print("V2")
plot_model_accuracy(train_history,'../plots/CNN_V2_training_history.jpg')

print("V1")
plot_model_loss(train_history, 'loss', '../plots/CNN_V1_training_loss.jpg')
print("V2")
plot_model_loss(train_history, 'loss', '../plots/CNN_V2_training_loss.jpg')

#score
score = model_CNN_V1.evaluate(test_data, test_label)
print('Test accuracy for CNN V1:', score[1])

#score
score2 = model_CNN_V2.evaluate(test_data, test_label)
print('Test accuracy: for CNN V2', score2[1])

#predictions
print("confusion matrix for CNN V1")
prediction = model_CNN_V1.predict(test_data)

prediction_class = np.argmax(prediction,axis=1)
prediction_label = np.argmax(test_label,axis=1)

#confusion matrix
mapping = lambda x:features_dict[x]
pred_class_to_feature = np.array([mapping(x) for x in prediction_class])
pred_label_to_feature = np.array([mapping(x) for x in prediction_label])
#pred_label_to_feature = prediction_label.map(lambda x:features_dict[x])

#confusion matrix
print(pd.crosstab(pred_label_to_feature,pred_class_to_feature,rownames=['actual'],colnames=['predicted']))

#predictions

print("confusion matrix for CNN V2")
prediction_2 = model_CNN_V2.predict(test_data)

prediction_class_2 = np.argmax(prediction_2,axis=1)
prediction_label_2 = np.argmax(test_label,axis=1)

#confusion matrix
mapping = lambda x:features_dict[x]
pred_class_to_feature_2 = np.array([mapping(x) for x in prediction_class_2])
pred_label_to_feature_2 = np.array([mapping(x) for x in prediction_label_2])
#pred_label_to_feature = prediction_label.map(lambda x:features_dict[x])

#confusion matrix
print(pd.crosstab(pred_label_to_feature_2,pred_class_to_feature_2,rownames=['actual'],colnames=['predicted']))
