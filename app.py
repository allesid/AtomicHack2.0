from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from base64 import b64encode
import io

from utils.bbox import draw_bboxes
from utils.visualizations import annotated_img_plotly_fig, annotated_img_plotly_meta
from defection_detector import DummyDetector, YoloModel

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import plotly
import json

from pathlib import Path

import pandas as pd

YOLO_PATH = Path('defection_detector\\model\\checkpoints\\YOLOv9c_50epochs.pt')

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

model = YoloModel(checkpoint_path=str(YOLO_PATH))

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
        predictions = model(img)
        log.info(predictions)

        # df = pd.DataFrame.from_records(predictions).drop(columns=['coords'])
        # log.info(df.to_string())
        # agg_df = df.groupby('label')['score'].mean()
        # log.info(agg_df.reset_index())

        # Make visaulization
        image = draw_bboxes(img, boxes=predictions, score=True)
        img_fig = annotated_img_plotly_fig(image)
        annotated_img = json.dumps(img_fig, cls=plotly.utils.PlotlyJSONEncoder)
        log.info(annotated_img)

        # Make prediction meta visualization
        dashboard_fig = annotated_img_plotly_meta(predictions)
        dashboard_json = json.dumps(dashboard_fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html', annotated_img_json=annotated_img, dashboard_json=dashboard_json)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
