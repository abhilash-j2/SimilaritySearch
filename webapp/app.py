from flask import Flask, render_template, url_for,send_file, make_response, request
import json
import io
from skimage.io import imread
import base64
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import numpy as np
import pickle
import os

import elasticsearch
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch import helpers


#from .imageProcess import get_random_image
plt.switch_backend('agg')
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', "jfif","tiff"])
cors = CORS(app, allow_headers='Content-Type', CORS_SEND_WILDCARD=True)

def load_model():
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = hub.load(module_handle)
    return module


def load_data():
    with open("./webapp/data/final_data", "rb") as f:
        df = pickle.load(f)
    return df


df = load_data()
model = load_model()
host = os.environ["ELASTIC_HOST"]
print(host)
es = Elasticsearch([{"host": host, "port":"9200"}])
es.ping()

def query_es(user_query):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['image_vector'])",
                "params": {"query_vector": user_query}
                        }
                    }
                }
    res = es.search(index='similarity-search',body={"from":0,"size": 5,"query": script_query,"_source":["picture_id","product_id"]})
    
    cards = []
    for dic in res['hits']['hits']:
        prod_id = dic['_source']['product_id']
        pic_id = dic['_source']['picture_id']
        score = dic['_score']
        picture = url_process_imgdict(df[df["picture_id"]==pic_id].to_dict("records")[0])
        cards.append({"prod_id" : prod_id, "pic_id" : pic_id, "score":score, "img" : picture})

    return cards

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
    user_query = imread(io.BytesIO(img_dict["picture"]["picture"]))
    user_query = get_image_vector(user_query)
    card_data = query_es(user_query)

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
            img_arr = img_arr[:,:,:3]
            features = get_image_vector(img_arr)
            card_data = query_es(features)

            return render_template('upload_result.html', mainimg = get_base64_from_img(img_arr), card_data = card_data )



def get_random_image():
    indx = np.random.randint(0,df.shape[0],size=1)
    d = df.iloc[indx,:].to_dict("records")[0]
    return d

def get_n_random_images(n):
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

def preprocess_img(img):
  img = tf.image.resize_with_pad(img,224,224)
  img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis,...]
  return img

def get_image_vector(image):
    img = image.copy()
    img = preprocess_img(img)
    features = model(img)
    features = np.squeeze(features)
    return features



if __name__ == '__main__':
   app.run(debug = True)