import logging
import os
from pathlib import Path
from typing import Literal, Dict

import numpy as np
from PIL import Image

from services.instance_segmentation.processor import OverlapPostProcessor

logging.basicConfig(level=logging.INFO)


class ObjectDetectionBasedInstanceSegmentationEngine:
    """
    Object detection based instance segmentation engine.

    Usage:
    engine = ObjectDetectionBasedInstanceSegmentationEngine()
    image = engine.load_input_image(path)
    predictions = engine.predict()
    object_id_map = engine.post_process()
    engine.export_object_id_map()
    """

    def __init__(self):
        self.image = None
        self.mask = None
        self.object_id_map = None

    def load_input_image(self, path: Path = None) -> Image:
        r"Load image to predict objects."
        self.image_path = path
        self.image_name = self.image_path.stem
        self.image = Image.open(path)

    def predict(self) -> Dict[str, np.ndarray]:
        r"Predict objects using object detection model."
        self.predictions = self._dummy_predict()
        return self.predictions

    def _dummy_predict(self) -> Dict[str, np.ndarray]:
        r"Dummy prediction that loads a precomputed prediction from a file."
        path = Path(os.path.join(self.image_path.parent, f"{self.image_name}.npz"))
        logging.info("Load dummy prediction from %s", path)
        return dict(np.load(str(path), allow_pickle=True))

    def post_process(
        self,
        overlap_strategy: Literal["distribute_by_confidence", "filter_by_confidence"] = "distribute_by_confidence",
    ):
        r"Return a 2d array of object_ids from the object detection predictions."
        processor = OverlapPostProcessor(self.image, self.predictions)
        self.object_id_map = processor.post_process(overlap_strategy)
        return self.object_id_map

    def export_object_id_map(self, path: Path = None):
        r"Export object_id_map to file with image name and object_id_map suffix. Defaults to image location."
        if path is None:
            path = self.image_path.with_suffix(".npy")
        np.save(str(path), self.object_id_map)
        logging.info("Export object_id_map to %s", path)
