FROM python:3.10.6-bullseye

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# apt-get update && apt-get install libgl1

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


#opencv-python = "4.6.0.66"
#uvicorn = {extras = ["standard"], version = "^0.22.0"}
#fastapi = "0.100.0"
#face-detection = "0.2.2"
#moviepy = "1.0.3"