import os
import io
import sys
import time
import datetime
import subprocess
import argparse
from skimage import io
from PIL import Image
import cv2
import numpy as np

import uuid
from utils import *

# python3 predict.py -p ./test --model_path ./models/mobilenetV2_model --gpu -1 --frame_rate 12 --denoise_borders --biggest_side 320



'''parser = argparse.ArgumentParser()
parser.add_argument('-p', '--data_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--gpu', type=int, default=-1, required=False)
parser.add_argument('--biggest_side', type=int, default=0, required=False)
parser.add_argument('--delay', type=int, default=7, required=False)
parser.add_argument('--frame_rate', type=int, default=12, required=False)
parser.add_argument('--denoise_borders', action='store_true')
args = parser.parse_args()
globals().update(vars(args))
'''
def convert_to_binary_mask(im):
    im=Image.fromarray(im)
    fill_color = (0,0,0)  # your new background color
    im = im.convert("RGBA")   # it had mode P after DL it from OP
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1]) # omit transparency
        im = background
    im=im.convert("RGB")
    im=np.array(im)  
    gray_mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_mask, 1, 255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def test(data_path,model_path):
    start = time.time()
    biggest_side=0
    #delay=7
    denoise_borders='storetrue'
    gpu=-1
    #frame_rate=12
    biggest_side = None if not biggest_side else biggest_side
    #delay = round(100/frame_rate + .5)

    trainer = Trainer(path=model_path, gpu=gpu) 
    if gpu < 0:
        torch.set_num_threads(2)
    trainer.load_state(mode="metric")
    trainer.model.eval()

    path=data_path
    img = cv2.imread(path,1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.uint8)
    out = trainer.predict_mask(img, biggest_side=biggest_side, denoise_borders=denoise_borders)
    name=str(uuid.uuid4())+'.png'        
    cv2.imwrite(name,out[0])

    print(" [INFO] %s ms. " % round((time.time()-start)*1000, 0))
    '''destination =os.path.join(path,'result',name)               
    save_image(destination,img)
    file_name='result/'+ name '''
    return out[0],name
img,path=test('./much_people.jpg','./mobilenetV2_model/mobilenetV2_model')
convert_to_binary_mask(path)
#!python /content/PicsArtHack-binary-segmentation/predict.py -p /content/t.jpg --model_path /content/my_model/mobilenetV2_model