import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
# model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# # model.add(BatchNormalization())

model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())

model.add(Flatten())

model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
model.load_weights('model11.h5')
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
face_image = cv2.imread("data/test/sad/im30.png")

try:
    face_locations = face_recognition.face_locations(face_image)
    top, right, bottom, left = face_locations[0]
    face_image = face_image[top:bottom, left:right]
except:
    pass

face_image = cv2.resize(face_image, (48, 48), cv2.INTER_AREA)
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

plt.imshow(face_image)
print(face_image.shape)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

prediction = model.predict(face_image)
maxindex = int(np.argmax(prediction))

print(emotion_dict[maxindex])


