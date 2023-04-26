import time

import face_detection
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

# Конфигурирование страницы
st.set_page_config(page_title="Пол_возраст", layout="wide", page_icon="random")

# Подгружаем модели при иницилаизации сервиса
img_size, detector, model, model_gender = get_models()

# Заголовок сервиса
st.header('Сервис по распознаванию пола и возраста')

# Виджет подгрузки контента
input_img = st.file_uploader("Choose video", type=["png", "jpg", "jpeg"])

# run only when user uploads file
if input_img is not None:

    file_bytes = np.asarray(bytearray(input_img.read()), dtype=np.uint8)
    input_img_cv = cv2.imdecode(file_bytes, 1)

    # Получаепм размеченное фото
    tagged_img = get_tagged_img(input_img_cv, logger, img_size, detector, model, model_gender)

    cv2.imwrite(f'users_detections/{input_img.name}', tagged_img)

    st.image(tagged_img, channels="BGR")