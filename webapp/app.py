from flask import Flask, render_template, url_for,send_file, make_response, request
import json
import io
from skimage.io import imread
import base64
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

import pandas as pd
import numpy as np
import pickle


#from .imageProcess import get_random_image
plt.switch_backend('agg')
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
cors = CORS(app, allow_headers='Content-Type', CORS_SEND_WILDCARD=True)

@app.route('/')
@app.route('/homepage')
def homepage():
    # return "Homepageview"
    return render_template('homepage.html')

@app.route('/randomizer')
def randomizer():
    # return "Homepageview"
    img_dict = get_random_image()
    mainimg = url_process_imgdict(img_dict)
    img_arr = get_n_random_images(5)
    img_arr = [ url_process_imgdict(img) for img in img_arr]
    imgscore=[1,2,3,4,5] 
    card_data = [{"img": img, "score" : score}  for img, score in zip(img_arr, imgscore)]

    return render_template('randomizer.html',pid=img_dict["product_id"]   ,
                             mainimg = mainimg, card_data = card_data)

def url_process_imgdict(img_dict):
    img = imread(io.BytesIO(img_dict["picture"]["picture"]))
    img = get_base64_from_img(img)
    return img 


@app.route('/randomImage', methods=['get','POST'])
def randomImage():
    img_dict = get_random_image()
    byte_io = io.BytesIO(img_dict["picture"]["picture"])
    # send_file(byte_io, attachment_filename='pic.png', mimetype='image/png')
    response = make_response(send_file(byte_io,mimetype='image/png'))
    response.headers['Content-Transfer-Encoding']='base64'
    return response

@app.route('/uploader')
def uploader():
    # return "Homepageview"
    return render_template('uploader.html')


def allowed_file(filename):
    """
    :param filename: 
    :return: 
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET','POST'])
@cross_origin(origins='*', send_wildcard=True)
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            byte_io = io.BytesIO()
            byte_io.write(file.read())
            byte_io.seek(0)

            img_arr = imread(byte_io)
            return render_template('randomizer.html',pid=0   , mainimg = get_base64_from_img(img_arr) )


def load_data():
    with open("./webapp/data/final_data", "rb") as f:
        df = pickle.load(f)
    return df

def get_random_image():
    df = load_data()
    indx = np.random.randint(0,df.shape[0],size=1)
    d = df.iloc[indx,:].to_dict("records")[0]
    return d

def get_n_random_images(n):
    df = load_data()
    indx = np.random.randint(0,df.shape[0],size=n)
    d = df.iloc[indx,:].to_dict("records")
    return d

def get_base64_from_img(img_arr):
    plt.imshow(img_arr)
    plt.axis("off")
    figfile = io.BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode("utf-8")
    return figdata_png





if __name__ == '__main__':
   app.run(debug = True)