import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
model.load_weights('model.h5')
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
face_image  = cv2.imread("TestImg/jjjj.png")


# face_locations = face_recognition.face_locations(face_image)
# top, right, bottom, left = face_locations[0]
# face_image = face_image[top:bottom, left:right]

#facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_image = cv2.resize(face_image, (48,48),cv2.INTER_AREA)
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

plt.imshow(face_image)
print(face_image.shape)
face_image = np.reshape(face_image, [1,face_image.shape[0], face_image.shape[1],1])


prediction = model.predict(face_image)
maxindex = int(np.argmax(prediction))

print(emotion_dict[maxindex])


