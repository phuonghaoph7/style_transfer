from flask_restful import Resource
from flask import send_file,Flask, request, render_template, send_from_directory, url_for, redirect
import flask
import uuid
from PIL import Image
import time
import numpy as np
from configs import path_to_apidir
import sys,os
from scipy.misc import imread, imresize, imsave
path=(os.path.dirname(path_to_apidir))
sys.path.append(path)
import style_transfer
from style_transfer.create_binary_mask import *
from scipy.misc import imread, imresize, imsave

class create_human_binary_mask(Resource):


    def __init__(self, mobile_path):
  
        self.mobile_path=mobile_path
      
    def post(self):
        mobile_path=self.mobile_path

        
        
        img = request.files['img']
    
        img = imread(img, mode='RGB')
        
        s=time.time()
        mask=human_segment(img,mobile_path)
        binary_mask=convert_to_binary_mask(mask)
        time_run=time.time()-s
        name = str(uuid.uuid4())+'.jpg'
        destination =os.path.join(path,'result',name)
       
        
        cv2.imwrite(destination,binary_mask)

        file_name='result/'+ name
        return {
			'tim': time_run,
			'result': url_for('static',filename=file_name),
            'size': img.shape  
            
		}