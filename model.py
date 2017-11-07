import csv
import os
import cv2
import numpy as np
import sklearn


def generator(samples, batch_size=32):
    correction = 0.2
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #print ('batch_sample: ', batch_sample)
                #name = './data-11-06-a/IMG/'+batch_sample[0].split('/')[-1]
                #center_image = cv2.imread(name)
                #center_angle = float(batch_sample[3])
                #images.append(center_image)
                #angles.append(center_angle)

                for i in range(3):
                    source_path = batch_sample[i]
                    #print ('i', i, source_path)
                    filename = source_path.split('/')[-1]
                    current_path = '../11-06-a/IMG/' + filename
                    srcBGR = cv2.imread(current_path)
                    image = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    if i == 0:
                        angle = float(batch_sample[3])
                        angles.append(angle)
                        #print ('IMAGE: ', current_path, angle)
                    else:
                        angle = float(batch_sample[3]) + (((-1)**i) * correction)
                        angles.append(angle)
                        #print ('IMAGE: ', current_path, angle)
                        

            augmented_images, augmented_angles = [], []
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #X_train = np.array(augmented_images)
            #y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)



lines = []
with open('../11-06-a/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader, None)  # skip the headers
  for line in reader:
    lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


#images = []
#measurements = []
correction = 0.2

print ('Num Lines:', len(lines))

#for line in lines:
#  for i in range(3):
#    source_path = line[i]
#    filename = source_path.split('/')[-1]
#    current_path = '../data/IMG/' + filename
#    image = cv2.imread(current_path)
#    images.append(image)
#    if i is 0:
#    	measurement = float(line[3])
#    	measurements.append(measurement)
#    else:
#    	measurement = float(line[3]) * ((-1)**i * correction)
#    	measurements.append(measurement)
#
#augmented_images, augmented_measurements = [], []
#for image,measurement in zip(images, measurements):
#  augmented_images.append(image)
#  augmented_measurements.append(measurement)
#  augmented_images.append(cv2.flip(image, 1))
#  augmented_measurements.append(measurement*-1.0)

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D


# compile and train the model using the generator function
batch_size=32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#		samples_per_epoch=len(train_samples),
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.fit_generator(train_generator, 
                samples_per_epoch = (len(train_samples)//batch_size)*batch_size,
		validation_data=validation_generator, 
            	nb_val_samples=len(validation_samples),
		nb_epoch=5)

model.save('model.h5')
