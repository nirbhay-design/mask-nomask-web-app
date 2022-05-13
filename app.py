from flask import *
import os
from torch_predictions import load_img, infer

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static','uploaded_imgs')

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict',methods= ['GET','POST'])
def predict():
    uploaded_file = request.files.get('image','')
    if not uploaded_file:
        return {"error":"upload file first"}
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'],"Inferenceimage.png")
    uploaded_file.save(upload_folder)
    img = load_img()
    prediction = infer(img)
    return render_template('home.html',img_name = uploaded_file.filename,img_path = upload_folder,predict=prediction,color = 'crimson' if prediction == "unmasked" else "green")

if __name__ == '__main__':
    app.run(debug = False)

