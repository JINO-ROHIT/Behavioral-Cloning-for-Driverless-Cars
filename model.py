import os
import csv
import cv2
import numpy as np
import sklearn
from keras.activations import relu
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

lines = []
with open('data2/driving_log.csv') as csvfile: #using data2
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#print(lines) #prints everything in the csv file row wise
def generate(total_images,batch_size=32):
    current_path = 'data2/IMG/'  #using data2
    while 1:
        total_images = sklearn.utils.shuffle(total_images) #randomly shuffling the images
        for offset in range(0,len(total_images),batch_size):
            batch_samples = total_images[offset:offset+batch_size] #to take 32 32 images each time
            #print(batch_samples)
           
            images = []
            angles = []
            for ind_sample in batch_samples:
                name1 = current_path + ind_sample[0].split('/')[-1] #the center images
                centre_only = cv2.imread(name1)
                if ind_sample[3] == 'steering':
                    continue
                centre_angle = float(ind_sample[3]) #taking the angles alone
                images.append(centre_only)
                angles.append(centre_angle)
                
                #flipping the images
                images.append(np.fliplr(centre_only))
                angles.append(-centre_angle)
                
                name2 = current_path + ind_sample[1].split('/')[-1] #these will extract the left images
                left_only = cv2.imread(name2)
                left_angle = float(ind_sample[3])+0.45 #the small correction factor
                images.append(left_only)
                angles.append(left_angle)
                
                name3 = current_path + ind_sample[2].split('/')[-1] #these are for the right images
                right_only = cv2.imread(name3)
                right_angle = float(ind_sample[3])-0.3 #the small correction factor
                images.append(right_only)
                angles.append(right_angle)
            #converting them to a numpy array    
            X_train = np.array(images) 
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train) #shuffling them
                
                
from sklearn.model_selection import train_test_split
train_samples, test_samples = train_test_split(lines, test_size=0.2,random_state=99) #random state for reproducibility,passing 20% for validation

train_generator = generate(train_samples, batch_size=32)
test_generator = generate(test_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda,Cropping2D

input_shape = (160,320,3)
my_model = Sequential()
my_model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
my_model.add(Cropping2D(cropping=((70,25), (0,0))))
my_model.add(Convolution2D(24, 5, 5, border_mode='valid', kernel_regularizer = regularizers.l2(0.001),activation='relu'))
my_model.add(MaxPooling2D()) 
my_model.add(Convolution2D(36, 5, 5, border_mode='valid', kernel_regularizer = regularizers.l2(0.001),activation='relu'))
my_model.add(MaxPooling2D())
my_model.add(Convolution2D(48, 5, 5, border_mode='valid', kernel_regularizer = regularizers.l2(0.001),activation='relu')) 
my_model.add(MaxPooling2D())
my_model.add(Convolution2D(64, 3, 3, border_mode='same', kernel_regularizer = regularizers.l2(0.001),activation='relu'))
my_model.add(Convolution2D(64, 3, 3, border_mode='valid', kernel_regularizer = regularizers.l2(0.001),activation='relu'))
my_model.add(MaxPooling2D())
my_model.add(Flatten())
my_model.add(Dense(80, kernel_regularizer = regularizers.l2(0.001)))
my_model.add(Dropout(0.5))
my_model.add(Dense(40, kernel_regularizer = regularizers.l2(0.001)))
my_model.add(Dropout(0.5))
my_model.add(Dense(16, kernel_regularizer = regularizers.l2(0.001)))
my_model.add(Dropout(0.5))
my_model.add(Dense(10, kernel_regularizer = regularizers.l2(0.001)))
my_model.add(Dense(1, kernel_regularizer = regularizers.l2(0.001)))
optimizer = Adam(lr = 0.0001)
my_model.compile(optimizer= optimizer, loss='mse')
                     
my_model.fit_generator(train_generator, 
            steps_per_epoch=len(train_samples)/32, 
            validation_data=test_generator, 
            validation_steps=len(test_samples)/32, 
            epochs=5, verbose=1)
                     
my_model.save('model.h5')
