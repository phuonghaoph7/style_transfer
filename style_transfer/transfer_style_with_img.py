import sys,os
from pathlib import Path
import time

from PIL import Image

import numpy as np
import tensorflow as tf
from itertools import product
from scipy.misc import imread, imresize, imsave
from style_transfer.sub_networks.Vgg import VGG


from .configs import config
from .image import load_image, prepare_image, load_mask, save_image
from .corall import corall
from .norm import adain
from .weights import open_weights
from .util import get_filename, get_params, extract_image_names_recursive



def load_model_mask(weights, decoder_weights, subnetwork,alpha):
  
    image = tf.compat.v1.placeholder(shape=(None,None,None,3), dtype=tf.float32)
    content = tf.compat.v1.placeholder(shape=(1,None,None,512), dtype=tf.float32)
    style = tf.compat.v1.placeholder(shape=(1,None,None,512), dtype=tf.float32)

    target = adain(content, style)
    weighted_target = target * alpha + (1 - alpha) * content
    #weighted_target = target
    with open_weights(weights) as w:
        encoder = subnetwork.build_subnetwork(image, w)['conv4_1']

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = subnetwork.build_decoder(weighted_target, w, trainable=False
               )
    else:
        decoder = subnetwork.build_decoder(weighted_target, None, trainable=False
            )

    return image, content, style, target, encoder, decoder


def transfer_with_mask(content_img, content_size, style_img,  mask,weights,decoder_weights,alpha): 
    s_time = time.time() 
    tf.compat.v1.disable_eager_execution()
    crop=config['crop']
    #preserve_color=config['preserve_color']
    preserve_color= None
    style_size=config['style_size']
 
    subnetwork=VGG()
    content_batch = [content_img]

    if mask.any():
            assert len(style_img) == 2, 'For spatial control provide two style images'
            style_batch = [style_img]
    elif len(style_img) > 1: # Style blending
            if not style_interp_weights:
                # by default, all styles get equal weights
                style_interp_weights = np.array([1.0/len(style_img)] * len(style_img))
            else:
                # normalize weights so that their sum equals to one
                #style_interp_weights = [float(w) for w in style_interp_weights.split(',')]
                style_interp_weights = np.array(style_interp_weights)
                style_interp_weights /= np.sum(style_interp_weights)
                assert len(style_img) == len(style_interp_weights), """--style and --style_interp_weights must have the same number of elements"""
            style_batch = [style_img]
    else:
            style_batch = style_img
   
    #print('Number of content images:', len(content_batch))
    #print('Number of style images:', len(style_batch))

    image, content, style, target, encoder, decoder=load_model_mask(weights, decoder_weights, subnetwork,alpha)
    

    with tf.compat.v1.Session() as sess:
        
        sess.run(tf.compat.v1.global_variables_initializer())
     

        for content_path, style_path in product(content_batch, style_batch):
            #content_name = get_filename(content_path)
            content_image = load_image(content_path, content_size, crop)

            if isinstance(style_path, list): # Style blending/Spatial control
                style_paths = style_path
                #style_name = '_'.join(map(get_filename, style_paths))

                # Gather all style images in one numpy array in order to get
                # their activations in one pass
                style_images = None
                for i, style_path in enumerate(style_paths):
                    style_image = load_image(style_path, style_size, crop='store_true')
                    #style_image = imresize(style_path, (style_size, style_size), interp='bilinear')
                    if preserve_color:
                        style_image = corall(style_image, content_image)
                    style_image = prepare_image(style_image,True)
                    if style_images is None:
                        shape = tuple([len(style_paths)]) + style_image.shape
                        style_images = np.empty(shape)
                    assert style_images.shape[1:] == style_image.shape, """Style images must have the same shape"""
                    style_images[i] = style_image
                    #style_features[i] = sess.run(encoder, feed_dict={image: style_images[i]})
                style_features = sess.run(encoder, feed_dict={image: style_images})
                    
                

                content_image = prepare_image(content_image,True)
                content_feature = sess.run(encoder, feed_dict={
                    image: content_image[np.newaxis,:]
                })

                if mask.any():
                    # For spatial control, extract foreground and background
                    # parts of the content using the corresponding masks,
                    # run them individually through AdaIN then combine
                    
                    _, h, w, c = content_feature.shape
                    content_view_shape = (-1, c)
                    mask_shape = lambda mask: (1, len(mask), c)
                    mask_slice = lambda mask: (mask,slice(None))

                    mask = load_mask(mask, h, w).reshape(-1)
                    fg_mask = np.flatnonzero(mask == 1)
                    bg_mask = np.flatnonzero(mask == 0)

                    content_feat_view = content_feature.reshape(content_view_shape)
                    content_feat_fg = content_feat_view[mask_slice(fg_mask)].reshape(mask_shape(fg_mask))
                    content_feat_bg = content_feat_view[mask_slice(bg_mask)].reshape(mask_shape(bg_mask))

                    style_feature_fg = style_features[0]
                    style_feature_bg = style_features[1]

                    target_feature_fg = sess.run(target, feed_dict={
                        content: content_feat_fg[np.newaxis,:],
                        style: style_feature_fg[np.newaxis,:]
                    })
                    target_feature_fg = np.squeeze(target_feature_fg)

                    target_feature_bg = sess.run(target, feed_dict={
                        content: content_feat_bg[np.newaxis,:],
                        style: style_feature_bg[np.newaxis,:]
                    })
                    target_feature_bg = np.squeeze(target_feature_bg)

                    target_feature = np.zeros_like(content_feat_view)
                    target_feature[mask_slice(fg_mask)] = target_feature_fg
                    target_feature[mask_slice(bg_mask)] = target_feature_bg
                    target_feature = target_feature.reshape(content_feature.shape)
                else:
                    # For style blending, get activations for each style then
                    # take a weighted sum.
                    target_feature = np.zeros(content_feature.shape)
                    for style_feature, weight in zip(style_features, style_interp_weights):
                        target_feature += sess.run(target, feed_dict={
                            content: content_feature,
                            style: style_feature[np.newaxis,:]
                        }) * weight
            else:
                #style_name = get_filename(style_path)
                style_image = load_image(style_path, style_size, crop)
                if preserve_color:
                    style_image = corall(style_image, content_image)
                style_image = prepare_image(style_image,True)
                content_image = prepare_image(content_image,True)
                style_feature = sess.run(encoder, feed_dict={
                    image: style_image[np.newaxis,:]
                })
                content_feature = sess.run(encoder, feed_dict={
                    image: content_image[np.newaxis,:]
                })
                target_feature = sess.run(target, feed_dict={
                    content: content_feature,
                    style: style_feature
                })

            output = sess.run(decoder, feed_dict={
                content: content_feature,
                target: target_feature
            })
    time_run=(time.time() - s_time)        
            
    return output[0],time_run

def transfer_no_mask(content_img, content_size, style_img,weights,decoder_weights,alpha): 
    s_time = time.time() 
    tf.compat.v1.disable_eager_execution()
    crop=config['crop']
    style_size=config['style_size']
    #preserve_color=config['preserve_color']
    preserve_color= None
    subnetwork=VGG()
    content_batch = [content_img]


    if len(style_img) > 1: # Style blending
                       
            style_interp_weights = np.array([1.0/len(style_img)] * len(style_img))
          
            style_batch = [style_img]
    else:
            style_batch = style_img
   
    #print('Number of content images:', len(content_batch))
    #print('Number of style images:', len(style_batch))

    image, content, style, target, encoder, decoder=load_model_mask(weights, decoder_weights, subnetwork,alpha)

    with tf.compat.v1.Session() as sess:
        
        sess.run(tf.compat.v1.global_variables_initializer())
     

        for content_path, style_path in product(content_batch, style_batch):
            #content_name = get_filename(content_path)
            content_image = load_image(content_path, content_size, crop)

            if isinstance(style_path, list): # Style blending/Spatial control
                style_paths = style_path
                #style_name = '_'.join(map(get_filename, style_paths))

                # Gather all style images in one numpy array in order to get
                # their activations in one pass
                style_images = None
                for i, style_path in enumerate(style_paths):
                    style_image = load_image(style_path, style_size, crop='store_true')
                    #style_image = imresize(style_path, (style_size, style_size), interp='bilinear')
                    if preserve_color:
                        style_image = corall(style_image, content_image)
                    style_image = prepare_image(style_image,True)
                    if style_images is None:
                        shape = tuple([len(style_paths)]) + style_image.shape
                        style_images = np.empty(shape)
                    assert style_images.shape[1:] == style_image.shape, """Style images must have the same shape"""
                    style_images[i] = style_image
                style_features = sess.run(encoder, feed_dict={
                    image: style_images
                })

                content_image = prepare_image(content_image,True)
                content_feature = sess.run(encoder, feed_dict={
                    image: content_image[np.newaxis,:]
                })

                
                    # For style blending, get activations for each style then
                    # take a weighted sum.
                target_feature = np.zeros(content_feature.shape)
                for style_feature, weight in zip(style_features, style_interp_weights):
                        target_feature += sess.run(target, feed_dict={
                            content: content_feature,
                            style: style_feature[np.newaxis,:]
                    }) * weight
            else:
                #style_name = get_filename(style_path)
                style_image = load_image(style_path, style_size, crop)
                if preserve_color:
                    style_image = corall(style_image, content_image)
                style_image = prepare_image(style_image,True)
                content_image = prepare_image(content_image,True)
                style_feature = sess.run(encoder, feed_dict={
                    image: style_image[np.newaxis,:]
                })
                content_feature = sess.run(encoder, feed_dict={
                    image: content_image[np.newaxis,:]
                })
                target_feature = sess.run(target, feed_dict={
                    content: content_feature,
                    style: style_feature
                })

            output = sess.run(decoder, feed_dict={
                content: content_feature,
                target: target_feature
            })
            
    time_run=(time.time() - s_time)       
    return output[0],time_run




'''content=Image.open('./input/content/avril.jpg')
style=Image.open('./input/style/antimonocromatismo.jpg')
style1=Image.open('./input/style/flower_of_life.jpg')
style2=Image.open('./input/style/mondrian.jpg')
styles=[]
styles.append(style)
styles.append(style1)
styles.append(style2)
weight=[1,1,1]'''
#img, time=multi_img(content,styles,1,512,vgg,decoder,weight)



            
