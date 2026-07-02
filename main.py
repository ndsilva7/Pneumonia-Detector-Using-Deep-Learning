from keras.models import load_model 
import cv2 
import numpy as np
import cvzone

np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)

class_names = ['Pneumonia', 'Normal']

img = cv2.imread('test/NORMAL/IM-0093-0001.jpeg')


image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

image = (image / 127.5) - 1


prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

texto1 = f"Class: {class_name}"
texto2 = f"Confidence Score: {str(np.round(confidence_score * 100))[:-2]} %"

print(texto1, texto2)


cvzone.putTextRect(img, texto1, (50,50), scale = 3)
cvzone.putTextRect(img, texto2, (50,100), scale = 3)



cv2.imshow('IMG', cv2.resize(img, (700,700)))
cv2.waitKey(0)



