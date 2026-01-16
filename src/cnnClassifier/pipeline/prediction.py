import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("model", "model.h5"))

        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = preprocess_input(test_image)

        predictions = model.predict(test_image)
        class_index = np.argmax(predictions, axis=1)[0]

        class_labels = {
            0: "Adenocarcinoma Cancer",
            1: "Large Cell Carcinoma",
            2: "Normal",
            3: "Squamous Cell Carcinoma"
        }

        prediction = class_labels[class_index]

        return [{"image": prediction}]