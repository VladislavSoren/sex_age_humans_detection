import cv2


# Функция получение новых ординат (чтобы бокс был квадратным)
def get_y_coords(yw1, yw2, xw1, xw2, img_h):

    k_sm = (yw2 - yw1) / (xw2 - xw1)
    shift = ((yw2 - yw1) - (yw2 - yw1) / k_sm) / 2
    yw1 = max(int(yw1 + shift), 0)
    yw2 = min(int(yw2 - shift), img_h - 1)
    # k_sm -> 1

    return yw1, yw2


# Функция получения сжатого изображения (степень сжатия зависит от размера)
def get_smaller_img(img):

    width_def = img.shape[1]
    height_def = img.shape[0]

    # Выбор кэфа масштабированивя в зависимости от габаритов изображения
    if (width_def > 3000) or (height_def > 3000):
        scale_percent = 40
    elif (width_def > 2000) or (height_def > 2000):
        scale_percent = 50
    elif (width_def > 1250) or (height_def > 1250):
        scale_percent = 60
    elif (width_def > 1000) or (height_def > 1000):
        scale_percent = 70
    elif (width_def > 800) or (height_def > 800):
        scale_percent = 80
    elif (width_def > 700) or (height_def > 700):
        scale_percent = 90
    else:
        return img

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized