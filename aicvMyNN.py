import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import csv

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
jsonStr = model.to_json()
# print("Model file converted to JSON:"+jsonStr)

resDict = {
    0: 'Arali',
    1: 'Arasamaram',
    2: 'Ashoka',
    3: 'Beetle',
    4: 'Curryleaf',
    5: 'Erukkampoo',
    6: 'Hibiscus',
    7: 'JackFruit',
    8: 'Lemon',
    9: 'Mango',
    10: 'Murungai',
    11: 'Nagalingapoo',
    12: 'Neem',
    13: 'Pudhina'
}


def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.array(x)
    array = model.predict(x)
    #print("Array:{}".format(array))
    result = array[0]
    #print("Result:{}".format(result))
    answer = np.argmax(result)
    if answer == 0:
        print(array[0][answer])
        print("Label: Arali")
    elif answer == 1:
        print(array[0][answer])
        print("Labels: Arasamaram")
    elif answer == 2:
        print(array[0][answer])
        print("Label: Ashoka")
    elif answer == 3:
        print(array[0][answer])
        print("Label: Beetle")
    elif answer == 4:
        print(array[0][answer])
        print("Label: Curryleaf")
    elif answer == 5:
        print(array[0][answer])
        print("Label: Erukkampoo")
    elif answer == 6:
        print(array[0][answer])
        print("Label: Hibiscus")
    elif answer == 7:
        print(array[0][answer])
        print("Label: JackFruit")
    elif answer == 8:
        print(array[0][answer])
        print("Label: Lemon")
    elif answer == 9:
        print(array[0][answer])
        print("Label: Mango")
    elif answer == 10:
        print(array[0][answer])
        print("Label: Murungai")
    elif answer == 11:
        print(array[0][answer])
        print("Label: Nagalingapoo")
    elif answer == 12:
        print(array[0][answer])
        print("Label: Neem")
    elif answer == 13:
        print(array[0][answer])
        print("Label: Pudhina")
    return answer


def cnnpredict(path):
    output = predict(path)
    with open('./AICV/predictedLabels.csv', 'a') as out:
        outwriter = csv.writer(out, delimiter=',')
        outwriter.writerow(['mycnn', path, resDict[output]])


path = input('Enter path of image : ')
cnnpredict(path)
