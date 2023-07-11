import logging
import time

import face_detection
from keras import backend as K

# импорт модуля распознования
from demo.SSRNET_model import SSR_net, SSR_net_general

import warnings
warnings.filterwarnings('ignore')


# Параметры логирования
logging.basicConfig(filename="models_init.log",
                    filemode="w",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

def get_models():

    # Initialize detector1
    time_tmp = time.time()
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    logging.info(f"Initialize detector: {time.time() - time_tmp}")

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
    logging.info(f"Building models: {time.time() - time_tmp}")

    return img_size, detector, model, model_gender