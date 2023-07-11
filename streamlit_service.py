import base64
import time
from pathlib import Path

import face_detection
from PIL import Image
from keras import backend as K

import cv2
import numpy as np

import streamlit as st
from streamlit.logger import get_logger

# импорт модуля распознования
from ssrnet_photo_detecting import get_tagged_img

from demo.SSRNET_model import SSR_net, SSR_net_general

import warnings
warnings.filterwarnings('ignore')


# Параметры логирования
logger = get_logger(__name__)


# Функция получения всех моделей (единожды при инициализации сервиса)
@st.cache_resource()
def get_models():

    # Initialize detector1
    time_tmp = time.time()
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    logger.info(f"Initialize detector: {time.time() - time_tmp}")

    # Set prediction mode (not learning)
    K.set_learning_phase(0)
    weight_file = "./pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "./pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"

    # fill hyper options
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1

    # Building models
    time_tmp = time.time()
    model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)
    model_gender = SSR_net_general(img_size, stage_num, lambda_local, lambda_d)()
    model_gender.load_weights(weight_file_gender)
    logger.info(f"Building models: {time.time() - time_tmp}")

    return img_size, detector, model, model_gender


@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)

    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


# Конфигурирование страницы
im = Image.open(Path.cwd()/'APP_icon'/'Иконка.png')
st.set_page_config(page_title="Пол_возраст", layout="wide", page_icon=im)

# Устанавливаем фон
set_png_as_page_bg(Path.cwd()/'APP_bg'/'Bg.jpg')

# Подгружаем модели при иницилаизации сервиса
img_size, detector, model, model_gender = get_models()

# Заголовок сервиса
st.header('Сервис по распознаванию пола и возраста')

url = 'https://t.me/VladislavSoren'
full_ref = f'<a href="{url}" style="color: #0d0aab">by FriendlyDev</a>'
st.markdown(f"<h2 style='font-size: 20px; text-align: right; color: black;'>{full_ref}</h2>", unsafe_allow_html=True)

# Виджет подгрузки контента
input_img = st.file_uploader("Choose video", type=["png", "jpg", "jpeg"])

# run only when user uploads file
if input_img is not None:

    image_input = input_img.read()
    file_bytes = np.asarray(bytearray(image_input), dtype=np.uint8)
    input_img_cv = cv2.imdecode(file_bytes, 1)

    # Получаепм размеченное фото
    tagged_img = get_tagged_img(input_img_cv, logger, img_size, detector, model, model_gender)

    cv2.imwrite(f'users_detections/{input_img.name}', tagged_img)

    st.image(tagged_img, channels="BGR")
else:
    st.header('preview')
    prev_img = cv2.imread('preview/prev_img.jpg')
    st.image(prev_img, channels="BGR")