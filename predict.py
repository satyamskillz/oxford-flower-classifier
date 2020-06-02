import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub


def process_image(image):
    image = tf.convert_to_tensor(image, np.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    return image


def predict(image_path, model_path, class_path, Top_k):
    with open(class_path, 'r') as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    img = Image.open(image_path)
    image = np.asarray(img)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)

    output = model.predict(image)

    probs, classes = tf.math.top_k(output, Top_k)
    classes += 1
    class_list = [class_names[str(i)] for i in classes[0].numpy()]
    for name, per in zip(class_list, probs.numpy()[0]):
        print("{} has the probility of {:0.5f}".format(name, per))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Flower Class')
    parser.add_argument("-f", "--imagePath", type=str, help="Enter image path")
    parser.add_argument("-m", "--modelPath", type=str, help="Enter h5 model path")
    parser.add_argument("-c", "--classPath", type=str, help="Enter class list path")
    parser.add_argument("-t", "--topk", type=int, default=5, help="Enter ")
    arg = parser.parse_args()
    print("Imgae Path-", arg.imagePath)
    print("Model Path-", arg.modelPath)
    print("Class Map Path-", arg.classPath)
    print("Imgae Path-", arg.topk)
    predict(arg.imagePath, arg.modelPath, arg.classPath, arg.topk)
