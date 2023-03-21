# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, flash
import os
from werkzeug.utils import secure_filename
from main import img_segmentation

#Path to save images/videos
UPLOAD_FOLDER = "static/img_vid/"

#App object
app = Flask(__name__, static_folder="static")

#To secure cookies, session is encrypted
app.secret_key = "segmentacja123"

#Folder for app used to save images/videos from user
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# '/' - is root directory, when getting into - looks for templates folder - get index.html
@app.route('/')
def index():
    return render_template('index.html')#,user_image="templates/tlo.jpg"

#Function to take submissions
@app.route('/',methods=['POST'])
def submit_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file'] # Take file given by user
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            #Take file and save it
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #Make prediction
            pred_img = img_segmentation(filename)
            
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            flash(full_filename)#Send original image to web app
            #flash(pred_img)#Send segmented image to web app
            return redirect('/')
            
if __name__ == "__main__":
    app.run()
