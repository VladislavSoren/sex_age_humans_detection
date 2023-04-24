
# Функция получение новых ординат (чтобы бокс был квадратным)
def get_y_coords(yw1, yw2, xw1, xw2, img_h):

    k_sm = (yw2 - yw1) / (xw2 - xw1)
    shift = ((yw2 - yw1) - (yw2 - yw1) / k_sm) / 2
    yw1 = max(int(yw1 + shift), 0)
    yw2 = min(int(yw2 - shift), img_h - 1)
    # k_sm -> 1

    return yw1, yw2