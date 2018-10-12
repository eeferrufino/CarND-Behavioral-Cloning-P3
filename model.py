
# coding: utf-8

# In[1]:

"""

For this projcet I imported the csv library to help in reading the steering angle and image data
I also imported the cv2(openCV) libraries to help in reading the image data
Numpy is used to convert the cv2 image objects into numpy arrays to support training, keras needs image data as a numpy array
Keras is the chosen deep learning library. I chose keras because we just learned about it and it seems like a very powerful library since very complex operations are just a single line of code.
Keras also makes the code very clean and easy to read/understand.

"""
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


# lines stores a row of csv data. Every lines element is a list the comma seperated data.
# lines[0] = center image data location
# lines[1] = left image data location
# lines[2] = right image data location
# lines[3] =
# lines[4] =
# lines[5] =
# lines[6] =

lines = []

# images is a buffer for image data

images = []

# measurements is a buffer for steering angle data

measurements = []

# Opens a hardcoded csv file and reads every row in the csv file into lines

with open('D:/P3_Data/super_mega_data_set/driving_log.csv') as csvfile2:
    reader2=csv.reader(csvfile2)
    for line in reader2:
        lines.append(line)

# Reads in the center, left, and right image data. For the left image data a correction is added to the steering angle measurement. For the right image data a correction is subtracted from the steering angle measurement.
# The order with which the addition/subtraction occur w.r.t correction is important becuase doing so incorrectly will associate the incorrect steering/image data and cause the network to learn to drive off the road rather than stay on the road.		
		
for line in lines:
    for i in range(3):
        orig_path = line[i]
        tokens = orig_path.split('\\')
        filename = tokens[-1]
        local_path = 'D:/P3_Data/super_mega_data_set/IMG/' + filename
        image = cv2.imread(local_path)
        images.append(image)
    correction = 0.15
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    

# Take the list of images and convert to numpy array and assign to a new list named X_train. X_train contain training image data.

X_train=np.array(images)

# Take the list of steering angle data and convert to a numpy array. y_train contains steerling angle data.

y_train=np.array(measurements)


# Quick check to make sure that length of images matches with length of measurements. A mismatch here could point to an error or bug in the above code and potential misalignment of image/steering angle data.
print(np.shape(images))
print(np.shape(measurements))

# The below model was based on the NVIDIA paper titled "End to End Learning for Self-Driving Cars"

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.3))
model.add(Dense(50))
model.add(Dropout(.3))
model.add(Dense(10))
model.add(Dropout(.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
    


# In[ ]:




