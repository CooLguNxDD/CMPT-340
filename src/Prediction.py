import gc

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16,InceptionV3
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# load metadata.csv
df = pd.read_csv('input/HAM10000_metadata.csv', delimiter=',')
df.dataframeName = 'HAM10000_metadata.csv'

def print_confusion_matrix(model):
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    score = model.evaluate(test_data, test_label)
    print('Test accuracy:', score[1])

    prediction = model.predict(test_data)
    prediction_class = np.argmax(prediction,axis=1)
    prediction_label = np.argmax(test_label,axis=1)

    mapping = lambda x:features_dict[x]
    pred_class_to_feature = np.array([mapping(x) for x in prediction_class])
    pred_label_to_feature = np.array([mapping(x) for x in prediction_label])
    matrix = confusion_matrix(pred_label_to_feature,pred_class_to_feature)
    matrix_display = ConfusionMatrixDisplay(matrix)
    fig,x = plt.subplots(figsize =(12,12))
    matrix_display.plot(ax=x)
    print(classification_report(pred_label_to_feature,pred_class_to_feature,target_names=labels))
def image_prediction(image_class):
    for images in image_class:
        test_image = image.load_img(images, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        result_VGG = model_VGG_16.predict(test_image)
        prediction_VGG = np.argmax(result_VGG, axis=1)
        print("VGG-16: predicted ", features_dict[prediction_VGG[0]])

        result_INC = model_INC.predict(test_image)
        prediction_INC = np.argmax(result_INC, axis=1)
        print("Inception V3: predicted ", features_dict[prediction_INC[0]])

        result_EFN = model_EF_Net.predict(test_image)
        prediction_EFN = np.argmax(result_EFN, axis=1)
        print("Efficient Net: predicted ", features_dict[prediction_EFN[0]])
        print()

# preprocess labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(df['dx'])
print(list(label_encoder.classes_))
df['label'] = label_encoder.transform(df["dx"])
print(df.sample(5))

# features to label
features_dict = {0:"akiex",1:"bcc",2:"bkl",3:"df",
                 4:"mel",5:"nv",6:"vasc"}

labels = ['akiec','bcc','bkl','df','mel','nv','vasc']

# Preprocess image
image_size = 150 #the size that the image will resize to
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('input/','*','*.jpg'))}
#image path
df['path'] = df['image_id'].map(lambda id: image_path.get(id))

#it takes time to process the image
df['image_64'] = df['path'].map(lambda path:Image.open(path).resize((image_size,image_size)))

# Image to array


df['image'] = df['image_64'].map(lambda image: np.asarray(image))
data = np.asarray(df['image'].to_list()).astype("short")
label_to_one_hot = to_categorical(df['label'], num_classes=7)

# Split train, test data
#80% data for training
#20% data for testing
train_data,test_data,train_label,test_label = train_test_split(data,label_to_one_hot,test_size=0.2,random_state=87,stratify=label_to_one_hot)

test_data = test_data/255.0

model_VGG_16 = load_model("model/VGG_16.h5")
model_EF_Net = load_model("model/Efficient_Net_model.h5")
model_INC = load_model("model/InceptionV3.h5")

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)

# Confusion Matrix and score



print_confusion_matrix(model_VGG_16)
print_confusion_matrix(model_INC)
print_confusion_matrix(model_EF_Net)


path_aki = 'demo\\ak'
path_bcc = 'demo\\bcc'
path_bkl = 'demo\\bkl'
path_df = 'demo\\df'
path_mel = 'demo\\mel'
path_nv_1 = 'demo\\nv_1'
path_vasc = 'demo\\vasc'

image_path_aki = [x for x in glob(os.path.join(path_aki,'*.jpg'))]
image_path_bcc = [x for x in glob(os.path.join(path_bcc,'*.jpg'))]
image_path_bkl = [x for x in glob(os.path.join(path_bkl,'*.jpg'))]
image_path_df = [x for x in glob(os.path.join(path_df,'*.jpg'))]
image_path_mel = [x for x in glob(os.path.join(path_mel,'*.jpg'))]
image_path_nv_1 = [x for x in glob(os.path.join(path_nv_1,'*.jpg'))]
image_path_vasc = [x for x in glob(os.path.join(path_vasc,'*.jpg'))]

image_class = image_path_vasc


