class Predict():
    def __init__(self):
        pass
    def __call__(self, img_path: str): # , prep_model: str
        """
        Detects objects in the image (Call from backend)

            Parameters:
                img_path (str): path to uploaded image or directory with images
            Returns:
                detection_results (List[List[dict]]): metadata for detection
                List of samples with list of dictionaries of defects data
        """
        # Open the image and pass to the real Object Detection model (please retunr coordinates in COCO format).
        # COCO: x y w h
        prep_model = 'best.pt'
        model = YOLO(model=prep_model)  # .pt
        result_yolo = model.predict(source=img_path)
        result = []
        for sample in result_yolo:
            res_sample = []
            ndef = sample.boxes.shape[0]
            for i in range(ndef):
                res_def = {'path':sample.path, 'label': int(sample.boxes.cls[i].item()),
                        'coords':sample.boxes.xywh[i].numpy().astype(int).tolist(), 'score': sample.boxes.conf[i].item()}
                res_sample.append(res_def)
            result.append(res_sample)

        # result = [
        #     {
        #         'coords': [262, 267, 37, 45],
        #         'label': 0,
        #         'score': 0.55
        #     },
        #     {
        #         'coords': [212, 143, 97, 55],
        #         'label': 1,
        #         'score': 0.96
        #     }
        # ]
        return result

