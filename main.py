import cv2
import numpy as np
import base64

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging

from functions_support import get_models, show_input_image, show_tagged_image
from ssrnet_photo_detecting import get_tagged_img

# from items_views import router as items_router
# from users.views import router as users_router

# Параметры логирования
logging.basicConfig(filename="models_predict.log",
                    filemode="w",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

app = FastAPI()

# Подгружаем модели при иницилаизации сервиса
img_size, detector, model, model_gender = get_models()


# app.include_router(
#     items_router,
#     prefix='/items',
# )
# app.include_router(
#     users_router,
#     prefix='/users',
# )


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

    # show_input_image()

    # image preprocessing
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    input_img_cv = cv2.imdecode(file_bytes, 1)

    # receiving tagged image
    tagged_img = get_tagged_img(input_img_cv, logging, img_size, detector, model, model_gender)

    # saving tagged image
    save_path = f'users_detections/{json_input.image_name}'
    cv2.imwrite(save_path, tagged_img)

    # show_tagged_image(tagged_img)

    # serialization
    json_out = {}
    json_out['tagged_img'] = image_bytes_to_str(save_path)

    return json_out


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=9988,
        # reload=True,
    )
