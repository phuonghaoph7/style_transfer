from scipy.misc import imread, imresize, imsave
import numpy as np


def load_image(image, size, crop):
    #image = imread(image, mode='RGB')
    #image=np.array(image)
    if crop=='store_true':
        image = central_crop(image)
    if size:
        image = scale_image(image, size)
    return image


def prepare_image(image, normalize=True):
    
    if normalize:
        image = image.astype(np.float32)
        image /= 255
    
    return image


def scale_image(image, size):
    "size specifies the minimum height or width of the output"
    #image=np.array(image)
    h,w,_= image.shape
    if h > w:
        image = imresize(image, (h*size//w, size), interp='bilinear')
    else:
        image = imresize(image, (size, w*size//h), interp='bilinear')
    return image


def central_crop(image):
    #image=np.array(image)
    h, w,_ = image.shape
    minsize = min(h, w)
    h_pad, w_pad = (h - minsize) // 2, (w - minsize) // 2
    image = image[h_pad:h_pad+minsize,w_pad:w_pad+minsize]
    return image


def save_image(filename, image):
    
    image *= 255
    image = np.clip(image, 0, 255)
    imsave(filename, image.astype(np.uint8))
    #cv2.imwrite(filename, image)

def load_mask(mask, h, w):
    #mask = imread(mask, mode='L')
    mask = imresize(mask, (h, w), interp='nearest')
    mask = mask.astype(np.uint8)
    mask[mask == 255] = 1
    return mask
