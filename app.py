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

    model = load_model('Model/AlexnetModel94-ori.h5', custom_objects={'Custom>SGD': SGD})
    imgi = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(imgi)[0]
    labels = (pred > 0.5).astype(int)
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    idx_pred = respon_model.index(max(respon_model))
    labels = list(class_list.keys())

    modelh = load_model('Model/DenseNet201Model96-ori.h5', custom_objects={'Custom>SGD': SGD})
    imgh = np.expand_dims(cv2.resize(img, modelh.layers[0].input_shape[0][1:3] if not modelh.layers[0].input_shape[1:3] else modelh.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    starth = time.time()
    predh = modelh.predict(imgh)[0]
    labelsh = (predh > 0.5).astype(int)
    print(labelsh)
    runtimesh = round(time.time()-starth,4)
    respon_modelh = [round(elem * 100, 2) for elem in predh]
    idx_predh = respon_modelh.index(max(respon_modelh))
    labelsh = list(class_list.keys())

    model1 = load_model('Model/ModelCNN1-ori.h5', custom_objects={'Custom>SGD': SGD})
    img1 = np.expand_dims(cv2.resize(img, model1.layers[0].input_shape[0][1:3] if not model1.layers[0].input_shape[1:3] else model1.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start1 = time.time()
    pred1 = model1.predict(img1)[0]
    labels1 = (pred1 > 0.5).astype(int)
    print(labels1)
    runtimes1 = round(time.time()-start1,4)
    respon_model1 = [round(elem * 100, 2) for elem in pred1]
    idx_pred1 = respon_model1.index(max(respon_model1))
    labels1 = list(class_list.keys())


    model2 = load_model('Model/ModelCNN2-ori.h5', custom_objects={'Custom>SGD': SGD})
    img2 = np.expand_dims(cv2.resize(img, model2.layers[0].input_shape[0][1:3] if not model2.layers[0].input_shape[1:3] else model2.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start2 = time.time()
    pred2 = model2.predict(img2)[0]
    labels2 = (pred2 > 0.5).astype(int)
    print(labels2)
    runtimes2 = round(time.time()-start2,4)
    respon_model2 = [round(elem * 100, 2) for elem in pred2]
    idx_pred2 = respon_model2.index(max(respon_model2))
    labels2 = list(class_list.keys())
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
                            labels1=labels1, 
                            probs1=respon_model1, 
                            model1='ModelCNN1', 
                            pred1=idx_pred1, 
                            run_time1=runtimes1,
                            labels2=labels2, 
                            probs2=respon_model2, 
                            model2='ModelCNN2', 
                            pred2=idx_pred2, 
                            run_time2=runtimes2,
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
        'Alexnet'           :   'Model/AlexnetModel94-ori.h5',
        'DenseNet201'       :   'Model/DenseNet201Model96-ori.h5',
        'ModelCNN1'         :   'Model/ModelCNN1-ori.h5',
        'ModelCNN2'         :   'Model/ModelCNN2-ori.h5',
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
    labels = (pred > 0.5).astype(int)
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
        'Alexnet'           :   'Model/AlexnetModel94-ori.h5',
        'DenseNet201'       :   'Model/DenseNet201Model96-ori.h5',
        'ModelCNN1'         :   'Model/ModelCNN1-ori.h5',
        'ModelCNN2'         :   'Model/ModelCNN2-ori.h5',
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
    labels = (pred > 0.5).astype(int)
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