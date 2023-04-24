import os
import cv2
import numpy as np
import argparse
from demo.SSRNET_model import SSR_net, SSR_net_general
import sys
import timeit
from moviepy.editor import *
from keras import backend as K
import time
import face_detection
from funcs.img_process import *

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def draw_results(detected,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot):

    souurce_img = input_img.copy()

    #for i, d in enumerate(detected):
    for i, (x,y,w,h,_) in enumerate(detected):

        x1 = int(x)
        y1 = int(y)
        x2 = int(w)
        y2 = int(h)

        w = int(w) - x
        h = int(h) - y

        # Увеличиваем бокс опираясь на величину ad
        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        yw1, yw2 = get_y_coords(yw1, yw2, xw1, xw2, img_h)

        # cv2.imshow("result", input_img[yw1:yw2 + 1, xw1:xw2 + 1, :])
        # cv2.waitKey(1)
        # time.sleep(2)

        # Вырезаем лицо и кладём на шаблон faces
        # resize_img = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        # cv2.imshow("result", resize_img)
        # cv2.waitKey(1)
        # time.sleep(2)

        '''
        Если вы увеличиваете изображение, лучше использовать интерполяцию INTER_LINEAR или INTER_CUBIC(лучше, но медленней). 
        Если вы сжимаете изображение, лучше использовать интерполяцию INTER_AREA 
        '''
        box_size = yw2 - yw1

        # Изменяем размер изображения применяя нужную интерполяцию
        if img_size > box_size:
            faces[i, :, :, :] = cv2.resize(souurce_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size),
                                           interpolation=cv2.INTER_CUBIC)
        else:
            faces[i, :, :, :] = cv2.resize(souurce_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size),
                                           interpolation=cv2.INTER_AREA)

        faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

        # cv2.imshow("result", input_img)
        # cv2.waitKey(1)
        # time.sleep(2)
        #
        # cv2.imshow("result", souurce_img)
        # cv2.waitKey(1)
        # time.sleep(2)
    
    start_time = timeit.default_timer()
    if len(detected) > 0:
        # predict ages and genders of the detected faces
        predicted_ages = model.predict(faces)
        predicted_genders = model_gender.predict(faces)
        

    # draw results
    for i, (x,y,w,h,_) in enumerate(detected):

        x1 = int(x)
        y1 = int(y)
        x2 = int(w)
        y2 = int(h)

        gender_str = 'male'
        if predicted_genders[i]<0.5:
            gender_str = 'female'

        label = "{},{}".format(int(predicted_ages[i]),gender_str)
        
        draw_label(input_img, (x1, y1), label)
    
    elapsed_time = timeit.default_timer()-start_time
    time_network = time_network + elapsed_time
    

    start_time = timeit.default_timer()

    elapsed_time = timeit.default_timer()-start_time
    time_plot = time_plot + elapsed_time

    return input_img,time_network,time_plot

def main():

    K.set_learning_phase(0) # make sure its testing mode
    weight_file = "./pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "./pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"

    path_source_img = './media_content/foto4/'
    save_path = './demo/img/'

    # Коэффициент расширения бокса
    ad = 0.5

    # Initialize detector
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    try:
        os.mkdir('demo/img')
    except OSError:
        pass

    # load model and weights
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)

    model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    model_gender.load_weights(weight_file_gender)
    
    # capture video
    paths_img = os.listdir(path_source_img)

    for img_idx, im_path in enumerate(paths_img):
        # get video frame
        input_img = cv2.imread(f'{path_source_img}{im_path}')

        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)

        time_detection = 0
        time_network = 0
        time_plot = 0

        # Детектируем лица
        start_time = timeit.default_timer()
        detections = detector.detect(input_img)

        elapsed_time = timeit.default_timer()-start_time
        time_detection = time_detection + elapsed_time

        faces = np.empty((len(detections), img_size, img_size, 3))

        input_img,time_network,time_plot = draw_results(detections,input_img,faces,ad,img_size,img_w,img_h,model,model_gender,time_detection,time_network,time_plot)
        cv2.imwrite(save_path+str(img_idx)+'.png',input_img)

        #Show the time cost (fps)
        print('avefps_time_detection:',1/time_detection)
        print('===============================')
        cv2.waitKey(1)
        time.sleep(3)
        


if __name__ == '__main__':
    main()
