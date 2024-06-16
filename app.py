from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from base64 import b64encode
import io
import uuid

from utils.bbox import draw_bboxes, LABELS
from utils.visualizations import annotated_img_plotly_fig, annotated_img_plotly_meta
from defection_detector import DummyDetector, YoloModel

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json

from pathlib import Path

import pandas as pd

from tqdm import tqdm
import shutil

import plotly.graph_objects as go

# TODO: dynamic upload
YOLO_PATH = Path('defection_detector\\model\\checkpoints\\YOLOv9c_50epochs.pt')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder
app.config['SAVE_BATCH'] = os.path.join('static', 'batch_results')
app.config['DOWNLOAD_PATH'] = os.path.join('downloads')
model = YoloModel(checkpoint_path=str(YOLO_PATH))

# TODO:Добавить норм описание к графикам

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    if request.method == 'POST':
        files = request.files.getlist("file")
  
        # Iterate for each file in the files List, and Save them
        total_images = len(files)
        total_predictions = []
        dataframes = []
        items = []

        uid = str(uuid.uuid4())
        save_results_to = os.path.join(app.config['SAVE_BATCH'], uid)
        os.makedirs(save_results_to)

        log.info(total_images)
        for file in tqdm(files):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD'], filename))
            img = os.path.join(app.config['UPLOAD'], filename)

            predictions = model(img)
            total_predictions.append(len(predictions))

            df = pd.DataFrame.from_records(predictions).drop(columns=['coords'])
            df['label'] = df['label'].map(LABELS)
            dataframes.append(df)
            items.append({'filename': filename, 'bboxes': predictions})

            image = draw_bboxes(img, boxes=predictions, score=True)
            image.save(os.path.join(save_results_to, filename))
        download_path = os.path.join(app.config['DOWNLOAD_PATH'], uid)
        with open(os.path.join(save_results_to, 'predictions.json'), 'w') as fd:
            json.dump(items, fd, indent=4)
        shutil.make_archive(download_path, 'zip', save_results_to)

        df = pd.concat(dataframes)
        agg_score_df = df.groupby('label')['score'].mean().reset_index()

        fig = make_subplots(rows=1, cols=2)
        fig.append_trace(go.Histogram(x=df["label"]), row=1, col=1)
        fig.append_trace(go.Bar(x=agg_score_df["label"], y=df["score"]), row=1, col=2)
        fig.update_layout(height=1200, width=1200, showlegend=False, title_text="Распрелеление классов (сверху), уверенность в каждом классе (снизу)")
        fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template(
            'dashboard.html',
            download_batch_link=f'{uid}.zip',
            countplot_json=fig
        )

@app.route('/back')
def back():
    return render_template('index.html')

@app.route('/downloads/<path:filepath>', methods=['GET', 'POST'])
def download(filepath):
    log.info(f'Download from: {filepath}')
    return send_from_directory('downloads', Path(filepath).name, as_attachment=True)

@app.route('/batch_form', methods=['GET', 'POST'])
def batch_form():
    return render_template('dashboard.html')

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

        # Make prediction meta visualization
        dashboard_fig = annotated_img_plotly_meta(predictions)
        dashboard_json = json.dumps(dashboard_fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html', annotated_img_json=annotated_img, dashboard_json=dashboard_json)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
