class DummyDetector:
    def __init__(self):
        pass
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
        result = [
            {
                'coords': [262, 267, 37, 45],
                'label': 0,
                'score': 0.55
            },
            {
                'coords': [212, 143, 97, 55],
                'label': 1,
                'score': 0.96
            }
        ]
        return result
