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
import scipy.misc
import uuid
from style_transfer.utils_human_segment import *

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
    mask= cv2.bitwise_not(mask)
    #name='mask.png'        
    #cv2.imwrite(name,mask)
    return mask

def human_segment(img,model_path,biggest_side=0):
    #im=Image.fromarray(im)
    #img=np.array(img)
    #img= scipy.misc.toimage(array)
    start = time.time()
    
    denoise_borders='storetrue' 
    biggest_side = None if not biggest_side else biggest_side
    trainer = Trainer(path=model_path, gpu=-1) 
  
    torch.set_num_threads(2)
    trainer.load_state(mode="metric")
    trainer.model.eval()

    
    #img = cv2.imread(path,1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.uint8)
    out = trainer.predict_mask(img, biggest_side=biggest_side, denoise_borders=denoise_borders)
    #name=str(uuid.uuid4())+'.png'        
    #cv2.imwrite(name,out[0])

    print(" [INFO] %s ms. " % round((time.time()-start)*1000, 0))
    
    return out[0]
#img,path=test('./much_people.jpg','./mobilenetV2_model/mobilenetV2_model')
#convert_to_binary_mask(path)