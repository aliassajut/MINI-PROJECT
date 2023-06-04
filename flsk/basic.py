from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
import cv2
import numpy as np
app= Flask(__name__)


from metrics import dice_loss, dice_coef, iou
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model=load_model('D:\downloads\model.h5')

def process_image(path):
    ori_x = cv2.imread(path)
    ori_x = cv2.resize(ori_x, (256, 256))
    x = ori_x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)[0] > 0.5
    y_pred = y_pred.astype(np.int32)
    save_image_path = 'D:\\downloads\\flsk\\static\\'+"output.png"
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred*255
    color = np.array([255, 0, 0]) 
    result = ori_x.copy()
    mask_indices = np.any(y_pred, axis=2)
    result[mask_indices] = color
    cv2.imwrite(save_image_path, result)
    return path

@app.route('/')


def index():
    return render_template('h.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        temp_file_path='D:\\downloads\\flsk\\static\\input.png'
        uploaded_file.save(temp_file_path)
        output= process_image(temp_file_path)       
       
        return render_template('result.html',output='output.png')
   

app.run()

