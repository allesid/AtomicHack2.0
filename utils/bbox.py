import cv2
from PIL import Image
import pybboxes as pbx

import logging
log = logging.getLogger(__name__)

LABELS = {
    0: 'hole1',
    1: 'hole2'
}
COLORS = [
    (89, 161, 197),
    (67, 161, 255)
]

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    bbox = pbx.convert_bbox(bbox=box['coords'], from_type='coco', to_type='voc', image_size=image.shape[:2])
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image, label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3,
            txt_color, thickness=tf, lineType=cv2.LINE_AA
        )

def draw_bboxes(image: str, boxes, score=True):
    image = cv2.imread(image)
    for box in boxes:
        label = LABELS[box['label']]
        if score:
            label += ' ' + str(round(100 * float(box['score']),1)) + "%"

        color = COLORS[box['label']]
        box_label(image, box, label, color)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
