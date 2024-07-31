from flask import Flask, request, send_file, render_template
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, save_img
from flask_cors import CORS
import os
import random
import numpy as np
import time
import subprocess as sp
import image_dehazer
import cv2
from PIL import Image

blur_to_hd = load_model('models/blur_to_hd.h5')
blur_to_hd.make_predict_function()

image_to_map = load_model('models/image_to_map_g_model.h5')
image_to_map.make_predict_function()

# fogg_removal = load_model('models/fogg_removal.h5')
# fogg_removal.make_predict_function()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'


def predict(filename, process_type):
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if process_type == 'fogg_removal':
        HazeImg = cv2.imread(input_path)
        HazeCorrectedImg, haze_map = image_dehazer.remove_haze(HazeImg)
        output_path.replace("jpg", "png")    
        print(output_path)    
        cv2.imwrite(output_path,HazeCorrectedImg )
    if process_type == 'object_removal':
        sp.Popen(['C:\\Users\\Aftab\\Documents\\python programs\\Photo Enhancer\\project\\Scripts\\python.exe', 'main.py', input_path])
    if process_type == 'image_to_map':
        size = (256, 256)
        sat_img = load_img(input_path, target_size=size)
        print(type(sat_img))
        sat_img = img_to_array(sat_img)
        print("printing the type of sat_img", type(sat_img))
        print("print karde bhai",sat_img)
        sat_img = (sat_img - 127.5) / 127.5
        input_image = []
        input_image.append(sat_img)
        input_image = np.array(input_image)
        print("printing the dimension of input_image np array", input_image.shape)
        gen_image = image_to_map.predict(input_image)
        print("before calc",gen_image[0, 1])
        gen_image = (gen_image+1)/2.0
        print("after cal", gen_image[0, 1])
        save_img(output_path, gen_image[0])
    if process_type == 'blur_to_hd':
        size = (256, 256)
        sat_img = load_img(input_path, target_size=size)
        sat_img = img_to_array(sat_img)
        sat_img = (sat_img - 127.5) / 127.5
        input_image = []
        input_image.append(sat_img)
        input_image = np.array(input_image)
        print(input_image.shape)
        gen_image = blur_to_hd.predict(input_image)
        gen_image = (gen_image+1)/2.0
        save_img(output_path, gen_image[0])


@ app.route("/")
def index():
    return render_template('index.html')


chars = "abcdefghijklmnopqrstuvwxyz"



@ app.route("/process_image", methods=['POST'])
def process_image():
    if 'image' in request.files:
        file = request.files['image']
        filename = ''.join(random.choice(chars)
                           for x in range(10)) + '.' + file.filename.split('.')[-1]
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        predict(filename, request.form['type'])
        return filename
    return "Hello"

@ app.route("/download")
def download():
    path = "static\\outputs\\" + request.args.get('path')
    print(path)
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
