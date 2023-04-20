import time
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads/'
st.set_option('deprecation.showfileUploaderEncoding', False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def predict_image(model, img):
    imgi = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(imgi)[0]
    labels = (pred > 0.5).astype(np.int)
    runtimes = round(time.time()-start, 4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    idx_pred = respon_model.index(max(respon_model))
    return labels, respon_model, runtimes, idx_pred

def model_comparison(img):
    class_list = ['leaf Blight', 'Brown Spot', 'Leaf Smut']
    with st.spinner('Loading Models...'):
        model1 = load_model('Model/AlexnetModel94-ori.h5')
        label1, prob1, rt1, pred1 = predict_image(model1, img)

        model2 = load_model('Model/DenseNet201Model96-ori.h5')
        label2, prob2, rt2, pred2 = predict_image(model2, img)

        model3 = load_model('Model/ModelCNN1-ori.h5')
        label3, prob3, rt3, pred3 = predict_image(model3, img)

        model4 = load_model('Model/ModelCNN2-ori.h5')
        label4, prob4, rt4, pred4 = predict_image(model4, img)

    st.markdown("## Model Comparison Results")
    st.image(img, use_column_width=True)

    cols = st.beta_columns(4)
    for i, col in enumerate(cols):
        col.header(f"Model {i+1}")
        col.write(f"Prediction: {class_list[locals()[f'pred{i+1}']]}")
        col.write(f"Probability: {max(locals()[f'prob{i+1}']):.2f}%")
        col.write(f"Runtime: {locals()[f'rt{i+1}']} seconds")

def main():
    st.title("Rice Leaf Disease Classification")
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Please select a page", ["Homepage", "Model Comparison"])

    if app_mode == "Homepage":
        st.markdown("## Home Page")
        st.write("This is the home page of the Rice Leaf Disease Classification App.")
        st.write("Please use the navigation bar on the left to select an option.")

    elif app_mode == "Model Comparison":
        st.markdown("## Model Comparison Page")
        st.write("Please upload an image to classify.")
        uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            model_comparison(img)

if __name__ == '__main__':
    main
