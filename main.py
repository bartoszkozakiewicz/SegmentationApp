# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.image as mpimg
from PIL import Image as im
import random
from collections import namedtuple
import cv2
from model_addons import label_to_color_image,DeepLabModel

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , 34 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

id2color = { label.id : np.asarray(label.color) for label in labels }



#Load models
unet_model = load_model("./model/UNETD15P5.h5", compile=False)


#IMAGES SEGMENTATION
def img_segmentation(filename,path):

    
    #Image preprocessing - for prediction
    img = mpimg.imread(path+filename)
    img = tf.image.resize(img,[256,256])
    img = tf.cast(img,tf.float32)
    img = img/255.
    img = np.asarray(img)
    img = img.reshape([1]+[256,256]+[3,])
    prediction = unet_model.predict(img)
    prediction = np.squeeze(np.argmax(prediction,axis=-1))
    
    #Image preprocessing - for visualization
    
    decoded_mask = np.zeros([prediction.shape[0],prediction.shape[1],3])
    for row in range(prediction.shape[0]):
        for col in range(prediction.shape[1]):
            decoded_mask[row,col,:] = id2color[prediction[row,col]]
            decoded_mask = decoded_mask.astype("uint8")    
    decoded_mask = im.fromarray(decoded_mask)   
    decoded_mask.save(f'{path}seg-{filename}.jpg')


#VIDEO SEGMENTATION

def preprocess_img(image):
  #img = tf.image.resize(img, [img_size,img_size])
  img = tf.cast(image,tf.float32)
  img = img/255.0
  return img

#CAPTURE VIDEO
def get_frames(video):
  frames = []
  # Open the video file
  video_capture = cv2.VideoCapture(video)
  # Check if the video file was opened successfully
  if not video_capture.isOpened():
      print('Error opening video file')
      exit()

  # Initialize frame counter
  frame_count = 0
  # Loop through the video frames
  while True:
      # Read the next frame from the video
      success, frame = video_capture.read()
      # Check if the frame was read successfully
      if not success:
          break
      frame_count+=1
      filename = f'frame_{frame_count}.jpg'
      cv2.imwrite(filename, frame)    
      frames.append(filename)
    
  return frames

#MAKE PREDICTIONS ON FRAMES FROM VIDEO
def eval_video(frames):
  images=[]
  decoded_frames=[]
  #Make predicted images (each frame from video)
  for img in frames:
    frame = mpimg.imread(img)
    frame = tf.image.resize(frame,[256,256])
    images.append(frame)
  eval_frames = tf.data.Dataset.from_tensor_slices(images)
  eval_frames = eval_frames.map(map_func=preprocess_img,num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=10).prefetch(tf.data.AUTOTUNE)
  pred_frames = unet_model.predict(eval_frames)
  decoded_mask = np.zeros([pred_frames[0].shape[0],pred_frames[0].shape[1],3])
  for pred in pred_frames:
    pred = np.squeeze(np.argmax(pred,axis=-1))
    for row in range(pred.shape[0]):
        for col in range(pred.shape[1]):
            decoded_mask[row,col,:] = id2color[pred[row,col]]
            decoded_mask = decoded_mask.astype("uint8")
    decoded_frames.append(decoded_mask)  
  return decoded_frames


def eval_video_2(frames,model):
  images=[]
  #Make predicted images (each frame from video)
  for frame in frames:
    image = im.open(frame)
    resized_im, seg_map = model.run2(image)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = cv2.addWeighted(np.array(resized_im), 0.6, seg_image, 0.9, 0.0)
    images.append(seg_image)
  return images


#CONVERT BACK TO VIDEO
def vid_conv(images,filename,path):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  fps=30

  vid = cv2.VideoWriter(f'{path}seg-{filename}.mp4',fourcc,fps,(256,256))
  for frame in images:
    vid.write(frame)
  cv2.destroyAllWindows()
  vid.release()

