import cv2
import numpy as np
from PIL import Image
import io

import base64

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging

from functions_support import get_models
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


class Data(BaseModel):
    user: str
    image: str


@app.post("/image")
def get_image(data: Data):
    # def get_image(data):
    print(data.user)
    print(type(data.image))

    image_bytes = base64.b64decode(data.image)
    print(type(image_bytes))

    image_data = image_bytes  # byte values of the image
    image = Image.open(io.BytesIO(image_data))
    print(type(image))
    image.show()


    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    print(type(file_bytes))
    input_img_cv = cv2.imdecode(file_bytes, 1)
    print(type(input_img_cv))


    # Получаепм размеченное фото
    tagged_img = get_tagged_img(input_img_cv, logging, img_size, detector, model, model_gender)
    print(type(tagged_img))
    img = Image.fromarray(tagged_img, 'RGB')
    print(type(img))
    img.show()

    return {
        "Image received"
    }


# @app.get("/hello")
# def hello(name: str = 'Soren', last_name: str = 'Sid'):
#     return {
#         f"message: Hello {name} {last_name}!"
#     }


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        # port=9988,
        reload=True,
    )
