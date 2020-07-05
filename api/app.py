from flask import Flask
from flask import Blueprint, current_app
from flask_restful import Api
from configs import path_to_apidir
import os,sys
import tensorflow as tf
apidir = path_to_apidir
path = os.path.join((os.path.dirname(path_to_apidir)))

import resources
from resources.transfer_image import transfer_image
from resources.transfer_mask import transfer_mask
from resources.download_image import download_image
from resources.create_human_binary_mask import create_human_binary_mask

my_project_path=(os.path.dirname(path_to_apidir))
print(my_project_path)
sys.path.append(my_project_path)

import style_transfer

app = Flask(__name__)
api =Api(app)

### The Routes for APIs response an object as JSON

def load_mask():
	my_project_path = os.path.join(path,'style_transfer')
	decoder_weights = os.path.join(my_project_path,'pre_train model','decoder_weights.h5')
	weights = os.path.join(my_project_path,'pre_train model','vgg19_weights_normalized.h5')	
	mobile_path=os.path.join(my_project_path,'mobilenetV2_model')	
	return  weights,decoder_weights,mobile_path
	
global encoder_mask, decoder_mask,mobile_path
encoder_mask, decoder_mask,mobile_path=load_mask()

api.add_resource(create_human_binary_mask, '/create_human_binary_mask',resource_class_kwargs={'mobile_path':mobile_path})

api.add_resource(transfer_image, '/transfer_image',resource_class_kwargs={'encoder_mask':encoder_mask, 'decoder_mask':decoder_mask})

api.add_resource(transfer_mask, '/transfer_mask',resource_class_kwargs={'encoder_mask':encoder_mask, 'decoder_mask':decoder_mask})

api.add_resource(download_image,'/static/result/<string:id>')



if __name__ == '__main__':

	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('-p')
	args = parser.parse_args()
	tmp = args.p
	app.run(host='0.0.0.0', port=tmp, debug=True)