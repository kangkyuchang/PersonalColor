import keras as keras
import numpy as np

model = keras.models.load_model('my_model.h5')
imgPath = "dataset/predict/2.jpeg"
img = keras.utils.load_img(imgPath, target_size=(416, 416))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(prediction)