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
from style_transfer.image import load_image, prepare_image, load_mask, save_image
from style_transfer import transfer_style_with_img
from style_transfer.transfer_style_with_img import transfer_with_mask,transfer_no_mask
from scipy.misc import imread, imresize, imsave

class transfer_image(Resource):

    def __init__(self, encoder_mask, decoder_mask):
  
        self.encoder_mask=encoder_mask
        self.decoder_mask=decoder_mask
      
    def post(self):
        encoder_mask=self.encoder_mask
        decoder_mask=self.decoder_mask
        

        img_style=[]
        
        content_img = request.files['content']

        style_img=request.files.getlist("style")
        content_size = int(request.form['size'])
        alpha = float(request.form['alpha'])
        content_img = imread(content_img, mode='RGB')
     
        
        for i in range(0,len(style_img)) :			         
                tmp = imread(style_img[i], mode='RGB')       
                img_style.append(tmp)
        img,time_run=transfer_no_mask(content_img, content_size, img_style, encoder_mask,decoder_mask,alpha)

        name = str(uuid.uuid4())+'.jpg'
        destination =os.path.join(path,'result',name)
       
        
        save_image(destination,img)

        file_name='result/'+ name
        return {
			'time': time_run,
			'result': url_for('static',filename=file_name),
            'size': img.shape  
            
		}