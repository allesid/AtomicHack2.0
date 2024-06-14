from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from base64 import b64encode
import io

from utils.bbox import draw_bboxes
from defection_detector import DummyDetector

import logging
log = logging.getLogger(__name__)

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        log.info('---> GOT FORM')
        # Get uploaded image
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        log.info(img)

        # Get predictions
        dm = DummyDetector()
        predictions = dm(img)

        # Make visaulization
        new_img = draw_bboxes(img, boxes=predictions, score=True)
        image_io = io.BytesIO()
        new_img.save(image_io, 'PNG')
        dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
        return render_template('index.html', img=dataurl)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
