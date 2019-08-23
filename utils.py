import cv2
import numpy as np

def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh

def normalize(img0, img_size):
        
    # Padded resize
    img, _, _, _ = letterbox(img0, height=img_size)
    # Normalize RGB
    # img = img[:, :, ::-1].transpose(2, 0, 1)
    n_img = np.ascontiguousarray(img, dtype=np.float32)
    n_img /= 255.0
    return n_img, img0