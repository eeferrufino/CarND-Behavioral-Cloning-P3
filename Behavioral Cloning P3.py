
# coding: utf-8

# In[1]:


import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
images = []
measurements = []


with open('D:/P3_Data/super_mega_data_set/driving_log.csv') as csvfile2:
    reader2=csv.reader(csvfile2)
    for line in reader2:
        lines.append(line)
#lines = lines[1:]        
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
    
augmented_images = []
augmented_measurements = []

#for image, measurement in zip(images, measurements)

X_train=np.array(images)
y_train=np.array(measurements)

print(np.shape(images))
print(np.shape(measurements))

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
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
    


# In[ ]:




