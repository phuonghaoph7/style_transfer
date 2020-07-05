import os
import yaml

config = None

path =os.path.dirname(os.path.abspath(__file__))

path=os.path.join(path, 'base.yaml')
#print(path)

with open(path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

