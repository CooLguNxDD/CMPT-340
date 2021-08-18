from helper import plot_model_accuracy, plot_model_accuracy

import os
import math
import random
from glob import glob
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dropout,Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential

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

features_dict = {0:"akiex",1:"bcc",2:"bkl",3:"df", 4:"mel",5:"nv",6:"vasc"}

#assign weight
#sum of the data
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

#image to array
df['image'] = df['image_64'].map(lambda image: np.asarray(image))
data = np.asarray(df['image'].to_list())
data = data.reshape(total_data,image_size*image_size*3).astype('float32')
data = data/255.0 #normalise the RGB value to [0...1]
label_to_one_hot = to_categorical(df['label'], num_classes=7)
#80% data for training
#20% data for testing
train_data,test_data,train_label,test_label = train_test_split(data,label_to_one_hot,test_size=0.2,random_state=87,stratify=label_to_one_hot)


# Model building
#setup MLP model
num_classes = 7
dimension = image_size*image_size*3

model_MLP_V1 = Sequential()

#input Dense layer of 64x64x3 image input:
#with normal distribution + relu activation
model_MLP_V1.add(Dense(units=2048,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V1.add(Dropout(0.1))

#hidden layer 2
model_MLP_V1.add(Dense(units=1024,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V1.add(Dropout(0.1))

#hidden layer 3
model_MLP_V1.add(Dense(units=512,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V1.add(Dropout(0.1))

#hidden layer 4
model_MLP_V1.add(Dense(units=256,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V1.add(Dropout(0.1))

#output Dense layer with 7 classes + softmax activation
model_MLP_V1.add(Dense(units=num_classes,kernel_initializer='normal',activation='softmax'))
model_MLP_V1.summary()

model_MLP_V1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# Train
try:
    model_MLP_V1.load_weights('../models/MLP_weight.h5')
except:
    print("new model_MLP_V1")

batch_size = 512
epochs = 200

train_history = model_MLP_V1.fit(
    x=train_data,y=train_label,
    epochs=epochs,
    batch_size = batch_size,
    validation_split=0.2,
    verbose=2,class_weight=class_weight)

model_MLP_V1.save_weights('../models/MLP_weight.h5')
model_MLP_V1.save('../models/MLP_V1.h5')

# apply oversampling instead of class weight
smote_sample = SMOTE(random_state=87)
train_data_oversample, train_label_oversample = smote_sample.fit_resample(train_data, train_label)

#setup MLP model V2
num_classes = 7
dimension = image_size*image_size*3

model_MLP_V2 = Sequential()

#input Dense layer of 64x64x3 image input:
#with normal distribution + relu activation
model_MLP_V2.add(Dense(units=2048,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V2.add(Dropout(0.2))

#hidden layer 2
model_MLP_V2.add(Dense(units=1024,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V2.add(Dropout(0.3))

#hidden layer 3
model_MLP_V2.add(Dense(units=512,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V2.add(Dropout(0.3))

#hidden layer 4
model_MLP_V2.add(Dense(units=256,input_dim=dimension,kernel_initializer='normal',activation='relu'))
model_MLP_V2.add(Dropout(0.3))

#output Dense layer with 7 classes + softmax activation
model_MLP_V2.add(Dense(units=num_classes,kernel_initializer='normal',activation='softmax'))
model_MLP_V2.summary()

model_MLP_V2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# Train
try:
    model_MLP_V2.load_weights('../models/MLP_V2_weight.h5')
except:
    print("new model_MLP_V2")

batch_size = 512
epochs = 200

train_history_2 = model_MLP_V2.fit(
    x=train_data_oversample,y=train_label_oversample,
    epochs=epochs,
    batch_size = batch_size,
    validation_split=0.2,
    verbose=2, shuffle=True)

model_MLP_V2.save_weights('../models/MLP_V2_weight.h5')
model_MLP_V2.save('../models/MLP_V2.h5')

# Model evaluation
#plot train history
print("MLP V1")
plot_model_accuracy(train_history,'../plots/MLP_V1_training_history.jpg')
print("MLP V2")
plot_model_accuracy(train_history,'../plots/MLP_V2_training_history.jpg')

# plot loss
print("MLP V1")
plot_model_loss(train_history, '../plots/MLP_V1_training_loss.jpg')
print("MLP V2")
plot_model_loss(train_history, '../plots/MLP_V2_training_loss.jpg')

#score
score = model_MLP_V1.evaluate(test_data, test_label)
print('Test accuracy:', score[1])

score_2 = model_MLP_V2.evaluate(test_data, test_label)
print('Test accuracy:', score_2[1])

#predictions
print("MLP V1")
prediction = model_MLP_V1.predict(test_data)

prediction_class = np.argmax(prediction,axis=1)
prediction_label = np.argmax(test_label,axis=1)

#confusion matrix
mapping = lambda x:features_dict[x]
pred_class_to_feature = np.array([mapping(x) for x in prediction_class])
pred_label_to_feature = np.array([mapping(x) for x in prediction_label])
#pred_label_to_feature = prediction_label.map(lambda x:features_dict[x])

#confusion matrix
print(pd.crosstab(pred_label_to_feature,pred_class_to_feature,rownames=['actual'],colnames=['predicted']))

print("MLP V2")
prediction_2 = model_MLP_V2.predict(test_data)

prediction_class_2 = np.argmax(prediction_2,axis=1)
prediction_label_2 = np.argmax(test_label,axis=1)

#confusion matrix
mapping = lambda x:features_dict[x]
pred_class_to_feature_2 = np.array([mapping(x) for x in prediction_class_2])
pred_label_to_feature_2 = np.array([mapping(x) for x in prediction_label_2])
#pred_label_to_feature = prediction_label.map(lambda x:features_dict[x])

#confusion matrix
print(pd.crosstab(pred_label_to_feature_2,pred_class_to_feature_2,rownames=['actual'],colnames=['predicted']))
