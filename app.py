# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, flash, get_flashed_messages
import os
from werkzeug.utils import secure_filename
from main import img_segmentation, get_frames, eval_video, vid_conv

#Path to save images/videos
UPLOAD_FOLDER = "static/img_vid/"

#App object
app = Flask(__name__, static_folder="static")

#To secure cookies, session is encrypted
app.secret_key = "segmentacja123"

#Folder for app used to save images/videos from user
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True

def cleanup(path):
    filesArray = os.listdir(path)
    for file in filesArray:
        os.remove(path+file)

# '/' - is root directory, when getting into - looks for templates folder - get index.html
@app.route('/')
def index():
    return render_template('index.html')#,user_image="templates/tlo.jpg"


@app.route('/cleanup')
def cl():
    cleanup(app.config['UPLOAD_FOLDER'])
    return redirect('/')#,user_image="templates/tlo.jpg"


#Function to take submissions
@app.route('/img',methods=['POST'])
def submit_image():
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
            img_segmentation(filename,app.config['UPLOAD_FOLDER'])
            
            flash(full_filename, 'img')#Send original image to web app
            flash(f'{app.config["UPLOAD_FOLDER"]}seg-{filename}.jpg', 'seg')#'static/img_vid/seg.jpg'
            
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
