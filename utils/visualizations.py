# Plotly
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json

import pandas as pd

from PIL import Image

from typing import List
from utils.bbox import LABELS

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VIS_W = 600

def annotated_img_plotly_fig(image: Image):
    img_width, img_height = image.size

    # Create a figure
    fig = go.Figure()

    # Add the image to the figure
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=img_height,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Set the axes properties
    fig.update_xaxes(visible=False, range=[0, img_width])
    fig.update_yaxes(visible=False, range=[0, img_height], scaleanchor="x")

    rescaling_coeff = 400 / max(img_width, img_height)

    # # Set the aspect ratio and margins
    fig.update_layout(
        width=int(img_width * rescaling_coeff),
        height=int(img_height * rescaling_coeff),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

def annotated_img_plotly_meta(predictions: List[dict]):
    df = pd.DataFrame.from_records(predictions).drop(columns=['coords'])
    df['label'] = df['label'].map(LABELS)
    agg_score_df = df.groupby('label')['score'].mean().reset_index()

    # 1. Class distribution
    # 2. Avg Score per class
    fig = make_subplots(rows=2, cols=1)
    fig.append_trace(go.Histogram(x=df["label"]), row=1, col=1)
    fig.append_trace(go.Bar(x=agg_score_df["label"], y=df["score"]), row=2, col=1)
    fig.update_layout(height=300, width=600, showlegend=False, title_text="Распрелеление классов (сверху), уверенность в каждом классе (снизу)")
    return fig
