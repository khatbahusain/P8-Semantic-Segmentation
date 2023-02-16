from flask import Flask, render_template, request, flash
import numpy as np
import re
import os
import pandas as pd
from collections import namedtuple
import cv2
import base64
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


IMG_SIZE = 192
NUM_CLASSES = 8

# Read the list of images
img_names = []
mask_names = []


for root, dirs, files in os.walk('static/data_val/val_mask'):
    for file in files:
        if file.endswith('labelIds.png'):
            mask_names.append(os.path.join(root, file))

for root, dirs, files in os.walk('static/data_val/val_img'):
    for file in files:
        if file.endswith('leftImg8bit.png'):
            img_names.append(os.path.join(root, file))


# Add the image and mask paths to data frame
# sort the list of images and masks
img_names.sort()
mask_names.sort()
df = pd.DataFrame({'img': img_names, 'mask': mask_names})


## 8 main categories
Label = namedtuple('Label', ['name', 'id', 'category', 'categoryId'])
labels = [    Label('unlabeled', 0, 'void', 0),    Label('ego vehicle', 1, 'void', 0),    Label('rectification border', 2, 'void', 0),    Label('out of roi', 3, 'void', 0),    Label('static', 4, 'void', 0),    Label('dynamic', 5, 'void', 0),    Label('ground', 6, 'void', 0),    Label('road', 7, 'flat', 1),    Label('sidewalk', 8, 'flat', 1),    Label('parking', 9, 'flat', 1),    Label('rail track', 10, 'flat', 1),    Label('building', 11, 'construction', 2),    Label('wall', 12, 'construction', 2),    Label('fence', 13, 'construction', 2),    Label('guard rail', 14, 'construction', 2),    Label('bridge', 15, 'construction', 2),    Label('tunnel', 16, 'construction', 2),    Label('pole', 17, 'object', 3),    Label('polegroup', 18, 'object', 3),    Label('traffic light', 19, 'object', 3),    Label('traffic sign', 20, 'object', 3),    Label('vegetation', 21, 'nature', 4),    Label('terrain', 22, 'nature', 4),    Label('sky', 23, 'sky', 5),    Label('person', 24, 'human', 6),    Label('rider', 25, 'human', 6),    Label('car', 26, 'vehicle', 7),    Label('truck', 27, 'vehicle', 7),    Label('bus', 28, 'vehicle', 7),    Label('caravan', 29, 'vehicle', 7),    Label('trailer', 30, 'vehicle', 7),    Label('train', 31, 'vehicle', 7),    Label('motorcycle', 32, 'vehicle', 7),    Label('bicycle', 33, 'vehicle', 7),    Label('license plate', -1, 'vehicle', 7)]


# create a dictionary that maps id to categoryId
dict_cat = {i.id : i.categoryId for i in labels}
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
    x = x.astype(np.float32) / 255.0
    return x

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    label_mask = np.zeros_like(mask)
    for k in dict_cat:
        label_mask[mask == k] = dict_cat[k] 
    label_mask = np.eye(NUM_CLASSES)[label_mask]
    return label_mask

def prediction(img):
    """
    Predicts a mask for a given image using an API.

    Parameters:
    img (str): The path to the image to be processed.

    Returns:
    mask (numpy.ndarray): A numpy array of shape (192, 192, 8) representing the predicted mask.
    """
    image = cv2.imread(img)
    # encode image as base64 string
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes)
    
    # send request
    json_content = {"image": img_base64.decode('utf-8')}
    response = requests.post("http://34.163.63.32:5000/", json=json_content)
    
    # decode the mask
    mask = base64.b64decode(response.json()["mask"])
    mask = np.frombuffer(mask, dtype=np.float32)
    mask = mask.reshape(192, 192, 8)
    return mask


####### Flask app #######

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'husain'


@app.route('/')
def index():
    return render_template('index.html', img_list=df['img'].values)

@app.route('/predicted', methods=['POST', 'GET'])
def predicted():
    if request.method == 'POST':
        path_img = request.form['option']
        index_df = df[df['img'] == path_img].index[0]
        path_mask = df['mask'][index_df]
        print(path_img)
        print(path_mask)
        img = read_image(path_img)
        mask = read_mask(path_mask)
        predicted_mask = prediction(path_img)
        plt.subplot(1, 3, 1)
        plt.title('image')
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.title('mask')
        plt.imshow(np.argmax(mask, axis=-1))
        plt.subplot(1, 3, 3)
        plt.title('predicted mask')
        plt.imshow(np.argmax(predicted_mask, axis=-1))
        plt.savefig('static/predicted.png')
        flash(path_img, 'prediction')
        flash(path_mask, 'prediction')
        flash("static/predicted.png", 'image')
        return render_template('index.html', img_list=df['img'].values)
