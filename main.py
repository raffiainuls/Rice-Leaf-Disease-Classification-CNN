import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/index.html', )

@app.route("/select")
def select():
    return render_template('/select.html', )

@app.route("/model_comparation")
def model_comparation():
    return render_template('/model_comparation.html', )

@app.route('/predict_model_comparation', methods=['POST'])
def predict_model_comparation():
    class_list = {'leaf Blight': 0, 'Brown Spot': 1, 'Leaf Smut': 2}
    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)

    model = load_model('Pratikum Modul 2/AlexnetModel94-ori.h5')
    imgi = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(imgi)[0]
    labels = (pred > 0.5).astype(np.int)
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    idx_pred = respon_model.index(max(respon_model))
    labels = list(class_list.keys())

    modelh = load_model('Pratikum Modul 2/DenseNet201Model96-ori.h5')
    imgh = np.expand_dims(cv2.resize(img, modelh.layers[0].input_shape[0][1:3] if not modelh.layers[0].input_shape[1:3] else modelh.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    starth = time.time()
    predh = modelh.predict(imgh)[0]
    labelsh = (predh > 0.5).astype(np.int)
    print(labelsh)
    runtimesh = round(time.time()-starth,4)
    respon_modelh = [round(elem * 100, 2) for elem in predh]
    idx_predh = respon_modelh.index(max(respon_modelh))
    labelsh = list(class_list.keys())
    return render_template('/result_model_comparation.html', 
                            img='temp.jpg',
                            labels=labels, 
                            probs=respon_model, 
                            model='Alexnet', 
                            pred=idx_pred, 
                            run_time=runtimes,
                            labelsh=labelsh, 
                            probsh=respon_modelh, 
                            modelh='DenseNet201', 
                            predh=idx_predh, 
                            run_timeh=runtimesh,
                            )
    # return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg')

@app.route("/select_image")
def select_image():
    return render_template('/select_image.html', )

@app.route('/predict_select_image', methods=['POST'])
def predict_select_image():
    chosen_model = request.form['select_model']
    file_image = request.form['file']
    model_dict = {
        'Alexnet'         :   'Pratikum Modul 2/AlexnetModel94-ori.h5',
        'DenseNet201'        :   'Pratikum Modul 2/DenseNet201Model96-ori.h5',
        }
    if chosen_model in model_dict:
        model = load_model(model_dict[chosen_model]) 
    else:
        model = load_model(model_dict[0])
    # file = request.files["file"]
    # file.save(os.path.join('static', 'temp.jpg'))
    pth_fl = 'static/main/images/sample_image/' + file_image
    img = cv2.cvtColor(np.array(Image.open(pth_fl)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    labels = (pred > 0.5).astype(np.int)
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result(chosen_model, runtimes, respon_model, 'main/images/sample_image/' + file_image)

@app.route("/about")
def about():
    return render_template('/about.html', )

@app.route('/predict', methods=['POST'])
def predict():
    chosen_model = request.form['select_model']
    model_dict = {
        'Alexnet'         :   'Pratikum Modul 2/AlexnetModel94-ori.h5',
        'DenseNet201'        :   'Pratikum Modul 2/DenseNet201Model96-ori.h5',
        }
    if chosen_model in model_dict:
        model = load_model(model_dict[chosen_model]) 
    else:
        model = load_model(model_dict[0])
    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    labels = (pred > 0.5).astype(np.int)
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg')

def predict_result(model, run_time, probs, img):
    class_list = {'leaf Blight': 0, 'Brown Spot': 1, 'Leaf Smut': 2}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)

if __name__ == '__main__':
	app.run(debug=True)