from ultralytics import YOLO
import torch

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class YoloModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(self.device)
        self.model = YOLO(model=checkpoint_path).to(self.device)
    def __call__(self, img_path: str):
        """
        Detects objects in the image (Call from backend)

            Parameters:
                img_path (str): path to uploaded image
            Returns:
                detection_results (List[dict]): metadata for detection
        """
        # Open the image and pass to the real Object Detection model (please reutnr coordinates in COCO format).
        # COCO: x y w h
        res = self.model.predict(img_path, save=False)
        result = []
        for i in range(len(res[0].boxes.xywh)):
            result.append(
                {
                    'coords': res[0].boxes.xywh[i].tolist(),
                    'label' :  int(res[0].boxes.cls[i].item()),
                    'score' : res[0].boxes.conf[i].item()
                }
            )

        return result
