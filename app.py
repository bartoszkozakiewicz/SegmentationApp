# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, flash, get_flashed_messages
import os
from werkzeug.utils import secure_filename
from main import img_segmentation, get_frames, eval_video, vid_conv
from model_addons import DeepLabModel, label_to_color_image, MODEL1, MODEL2
from PIL import Image
import numpy as np

#Path to save images/videos
UPLOAD_FOLDER = "static/img_vid/"

#App object
app = Flask(__name__, static_folder="static")

#To secure cookies, session is encrypted
app.secret_key = "segmentacja123"

#Folder for app used to save images/videos from user
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True

model_name = ""

def cleanup(path):
    filesArray = os.listdir(path)
    for file in filesArray:
        os.remove(path+file)

# '/' - is root directory, when getting into - looks for templates folder - get index.html
@app.route('/', methods=['POST','GET'])
def index():
    
    # Zapisanie nazwy modelu jako zmiennej globalnej
    global model_name
    if request.method == 'POST':
        model_id = request.form["model"]
        if model_id == 'model1':
            model_name='model1'
        elif model_id == 'model2':
            model_name='model2'
        elif model_id == 'model3':
            model_name = 'model3'
        else:
            return 'Invalid model selected'        
            
    return render_template('index.html')


@app.route('/cleanup')
def cl():
    cleanup(app.config['UPLOAD_FOLDER'])
    return redirect('/')#,user_image="templates/tlo.jpg"


#Function to take submissions
@app.route('/img',methods=['POST'])
def submit_image():
    
    global model_name
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file'] # Take file given by use
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            #Take file and save it
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(full_filename)
            
            #Make prediction - UNET
            if model_name == "model1":
                img_segmentation(filename,app.config['UPLOAD_FOLDER'])
                        
                flash(full_filename, 'img')#Send original image to web app
                flash(f'{app.config["UPLOAD_FOLDER"]}seg-{filename}.jpg', 'seg')#'static/img_vid/seg.jpg
                return redirect('/')
            
            #Make prediction - 65
            elif model_name == "model2":
                image = Image.open(full_filename)
                resized_im, seg_map = MODEL2.run(image)
                seg_image = label_to_color_image(seg_map).astype(np.uint8)
                
                flash(full_filename, 'img')#Send original image to web app
                flash(f'{app.config["UPLOAD_FOLDER"]}seg-model2-{filename}.jpg', 'seg')#'static/img_vid/seg.jpg
                return redirect('/')
            
            #Make prediction - Xception71
            elif model_name == "model3":
                image = Image.open(full_filename)
                resized_im, seg_map = MODEL1.run(image)
                seg_image = label_to_color_image(seg_map).astype(np.uint8)
                
                flash(full_filename, 'img')#Send original image to web app
                flash(f'{app.config["UPLOAD_FOLDER"]}seg-model3-{filename}.jpg', 'seg')#'static/img_vid/seg.jpg
                return redirect('/')
            
@app.route('/video',methods=['POST'])
def submit_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file'] # Take file given by use
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            #Take file and save it
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(full_filename)
            
            #Make prediction
            frames2 = get_frames(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            pred_images2 = eval_video(frames2)
            vid_conv(pred_images2, filename,app.config['UPLOAD_FOLDER'])
            
            flash(full_filename, 'video')#Send original video to web app
            flash(f'{app.config["UPLOAD_FOLDER"]}seg-{filename}.mp4', 'video_seg')
            
            return redirect('/')        
            
if __name__ == "__main__":
    app.run()
