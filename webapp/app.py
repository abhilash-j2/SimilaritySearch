from flask import Flask, render_template, url_for,send_file, make_response, request
import json
import io
from skimage.io import imread
import base64
import matplotlib.pyplot as plt
# from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import requests

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


def load_data():
    with open("./webapp/data/final_data", "rb") as f:
        df = pickle.load(f)
    return df

with open("./webapp/data/lookup_df", "rb") as f:
        lookup_df = pickle.load(f)
# print(lookup_df.head())
df = load_data()
host = os.environ["ELASTIC_HOST"]
print(host)

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
    
    cards = []
    try:    
        with Elasticsearch([{"host": host, "port":"9200"}]) as es: 
            res = es.search(index='similarity-search',body={"from":0,"size": 5,"query": script_query,"_source":["picture_id","product_id"]})
            for dic in res['hits']['hits']:
                prod_id = dic['_source']['product_id']
                pic_id = dic['_source']['picture_id']
                score = dic['_score']
                picture = url_process_imgdict(df[df["picture_id"]==pic_id].to_dict("records")[0]["picture"])
                cards.append({"prod_id" : prod_id, "pic_id" : pic_id, "score":score, "img" : picture})

    except Exception as e:
        print(e)
        

    

    return cards

@app.route('/')
@app.route('/homepage')
def homepage():
    # return "Homepageview"
    return render_template('homepage.html')

def get_image_for_pid(pid):
    return df[df["picture_id"] == pid].to_dict("records")[0]

def get_random_neighbours():
    master_pi = np.random.choice(list(set(lookup_df["master_pi"])), size=1)[0]
    # print(f"== master pi {master_pi}")
    nns = lookup_df[lookup_df["master_pi"] == master_pi]
    return nns


def construct_card_data(nns):
    cards = []
    nns = nns.merge(df, left_on="similar_pi", right_on="picture_id", how="inner")
    for row in nns.itertuples():
        cards.append({"prod_id" : row.product_id, 
                      "pic_id" : row.picture_id, 
                      "score":row.similarity, 
                      "img" : url_process_imgdict(row.picture)})
    return cards

@app.route('/randomizer')
def quick_randomizer():
    nns = get_random_neighbours()
    img_dict = get_image_for_pid(list(set(nns.master_pi))[0])
    mainimg = url_process_imgdict(img_dict["picture"])
    card_data = construct_card_data(nns)
    return render_template('randomizer.html',pid=img_dict["product_id"]   ,
                             mainimg = mainimg, card_data = card_data)


def randomizer():
    # return "Homepageview"
    img_dict = get_random_image()
    mainimg = url_process_imgdict(img_dict)
    with io.BytesIO(img_dict["picture"]["picture"]) as f:
        user_query = imread(f)
    user_query = get_image_vector(user_query)
    card_data = query_es(user_query)

    return render_template('randomizer.html',pid=img_dict["product_id"]   ,
                             mainimg = mainimg, card_data = card_data)

def url_process_imgdict(img_dict):
    with io.BytesIO(img_dict["picture"]) as f:
        img = imread(f)
    img = get_base64_from_img(img)
    return img 


@app.route('/randomImage', methods=['get','POST'])
def randomImage():
    img_dict = get_random_image()
    with io.BytesIO(img_dict["picture"]["picture"]) as f:
        byte_io = imread(f)
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
            with io.BytesIO() as byte_io:
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
    with io.BytesIO() as figfile:
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode("utf-8")
    return figdata_png


MODEL_URL = os.environ["IMAGE_MODEL_URL"]
def get_image_vector(image):
    dictToSend = {"image":image.tolist()}
    res = requests.post(MODEL_URL, json=dictToSend)
    # print (('response from server:',res.text))
    dictFromServer = res.json()
    return dictFromServer["features"]

if __name__ == '__main__':
   app.run(debug = True)