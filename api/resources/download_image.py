from flask_restful import Resource
from flask import Flask, request, render_template,send_from_directory
import flask
import sys,os
from configs import path_to_apidir

path=os.path.dirname(path_to_apidir)
path=os.path.join(path,'result')

def send_image(filename):
        return send_from_directory(path, filename)

class download_image(Resource):

	def get(self, id):
            return send_image(id)