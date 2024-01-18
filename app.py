from flask import Flask, render_template, request
import torch
import json
from werkzeug.utils import secure_filename
from model.model import predict
from helper.load_model import load_checkpoint
import os

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
checkpoint_path = "checkpoint_new.pth"
model = load_checkpoint(checkpoint_path)
# model.to(device);


@app.route("/", methods=["GET", "POST"])
def imputImage():
    if request.method == "POST":
        image_file = request.files['image']
        print("flower",image_file.filename)
        if image_file:
            image_folder = os.path.join('uploads', secure_filename(image_file.filename) )
            print("image_folder",image_folder)
            image_file.save(image_folder)
            probs, classes = predict(image_file, model, 5)
            print(probs, classes)

            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
                names = [cat_to_name[key] for key in classes]
                name = names[0]
                print("Class name:")
                print(names)
            return render_template("index.html", prediction = name)
    return render_template("index.html", prediction = None)

if __name__ == "__main__":
    app.run(debug=True)
