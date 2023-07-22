import os

import cv2
import numpy as np
import base64

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging

from functions_support import get_models
from ssrnet_photo_detecting import get_tagged_img

# Параметры логирования
logging.basicConfig(filename="models_predict.log",
                    filemode="w",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

app = FastAPI()

# Подгружаем модели при иницилаизации сервиса
img_size, detector, model, model_gender = get_models()


# Обработка запроса на корень нашего сервиса
@app.get("/")
def index():
    return {
        "message: Index!"
    }


class PredictRequest(BaseModel):
    user: str
    image: str
    image_name: str


def image_bytes_to_str(im_path):
    with open(im_path, mode='rb') as file:
        image_bytes = file.read()
    image_str = base64.encodebytes(image_bytes).decode('utf-8')
    return image_str


@app.post("/image")
def get_image(json_input: PredictRequest):
    #  deserialization
    image_bytes = base64.b64decode(json_input.image)

    # image preprocessing
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    input_image_cv = cv2.imdecode(file_bytes, 1)

    # receiving tagged image
    tagged_image = get_tagged_img(input_image_cv, logging, img_size, detector, model, model_gender)

    # saving tagged image
    if not os.path.exists('users_detections'): os.makedirs('users_detections')
    save_path = f'users_detections/{json_input.image_name}'
    cv2.imwrite(save_path, tagged_image)

    # serialization
    json_out = {}
    json_out['tagged_image'] = image_bytes_to_str(save_path)

    return json_out


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=9988,
        # reload=True,
    )
