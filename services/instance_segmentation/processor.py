import logging
from typing import Literal, Tuple, Dict, List, Union

import numpy as np
from PIL import Image
from skimage.transform import resize

logging.basicConfig(level=logging.INFO)


class OverlapPostProcessor:
    """
    Post process instance segmentation predictions. Bounding box duplicates
    get removed and overlapping predictions get distributed or filtered
    by confidence.
    :param image: PIL image
    :param predictions: dict with keys "boxes", "labels", "scores", "masks"
    :param overlap_strategy:  "distribute_by_confidence", "filter_by_confidence"
    :return: 2d array of object_ids

    "distribute_by_confidence": Distribute overlapping predictions. The mask
                                of the prediction with the highest confidence
                                gets the overlapping area.
    "filter_by_confidence": Filter overlapping predictions by confidence and
                            keep the one with the highest confidence and remove
                            the other one.

    Usage:
    processor = OverlapPostProcessor(image, predictions)
    object_id_map = processor.post_process(overlap_strategy="distribute_by_confidence")
    """

    def __init__(self, image: Image, predictions: Dict[str, np.ndarray]):
        self.image = image
        self.prediction_dict = predictions
        self.len_predictions = len(self.prediction_dict["boxes"])
        self.overlap_idxs = None

    def post_process(
        self,
        overlap_strategy: Literal["distribute_by_confidence", "filter_by_confidence"],
        soft_mask_threshold: float = 0.5,
    ) -> np.ndarray:
        r"Post process predictions."
        self._remove_duplicates()
        overlaps = self._find_overlaps()
        return self._handle_overlaps(overlaps, overlap_strategy, soft_mask_threshold)

    @property
    def image_size(self) -> Tuple:
        r"Return image size as tuple."
        return self.image.size

    def _remove_instances_in_predictions(self, rows: Union[int, List[int], np.ndarray, List[np.ndarray]]):
        r"Remove row in prediction."
        for key, value in self.prediction_dict.items():
            self.prediction_dict[key] = np.delete(value, rows, axis=0)

    def _deduplicate_predictions(self):
        r"Remove duplicate predictions."
        _, unique_idxs = np.unique(self.prediction_dict["boxes"], axis=0, return_index=True)
        for idx in range(self.len_predictions):
            if idx not in unique_idxs:
                self._remove_instances_in_predictions(idx)

    def _handle_overlaps(
        self,
        overlaps,
        overlap_strategy: Literal["distribute_by_confidence", "filter_by_confidence"],
        soft_mask_threshold: float,
    ):
        r"Handle overlapping predictions."
        if overlap_strategy == "distribute_by_confidence":
            self._distribute_by_confidence_if_overlaps_exist(overlaps)
        elif overlap_strategy == "filter_by_confidence":
            self._filter_by_confidence_if_overlaps_exist(overlaps)
        else:
            logging.warning("Overlap_strategy {overlap_strategy} not supported")
            raise ValueError(f"Overlap_strategy {overlap_strategy} not supported")
        return self._create_object_id_map(self.prediction_dict, soft_mask_threshold)

    def _find_overlaps(self) -> np.ndarray:
        r"Find overlapping predictions."

        def boxes_are_overlapping(box1, box2) -> bool:
            r"Check if bounding boxes overlap using box coordinates."
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2
            return x1 < x4 and x3 < x2 and y1 < y4 and y3 < y2

        overlaps = []
        for idx1, bbox1 in enumerate(self.prediction_dict["boxes"]):
            for idx2, bbox2 in enumerate(self.prediction_dict["boxes"]):
                if idx1 != idx2 and boxes_are_overlapping(bbox1, bbox2):
                    overlaps.append((idx1, idx2))
        self.overlap_idxs = np.array(overlaps)
        return self.overlap_idxs

    def _remove_duplicates(self):
        r"Remove duplicate predictions. Find boxes with identical coordinates."
        logging.info("Remove duplicate predictions")
        self._deduplicate_predictions()
        return self.prediction_dict

    def _scale_mask_to_box(self, mask, box):
        r"Scale 2d mask to bounding box size with scikit-image resize."
        mask = resize(mask, (box[3] - box[1], box[2] - box[0]), anti_aliasing=False)
        return mask

    def _merge_boxes(box1, box2):
        r"Merge two bounding boxes."
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return np.array([min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)])

    def _distribute_by_confidence_if_overlaps_exist(self, overlaps):
        r"Distribute overlapping predictions by confidence."
        if len(overlaps) == 0:
            logging.info("No overlaps found")
        else:
            self._distribute_by_confidence()

    def _distribute_by_confidence(self):
        r"Sort overlaps by confidence and distribute them."
        idxs = np.argsort(self.prediction_dict["scores"])
        for key, value in self.prediction_dict.items():
            self.prediction_dict[key] = value[idxs]

    def _filter_by_confidence_if_overlaps_exist(self, overlaps):
        r"Filter overlapping predictions by confidence."
        if len(overlaps) == 0:
            logging.info("No overlaps found.")
        else:
            self._filter_by_confidence(overlaps)

    def _filter_by_confidence(self, overlaps):
        r"Filter overlapping predictions by confidence."
        idxs_to_remove = []
        for idx1, idx2 in overlaps:
            score1 = self.prediction_dict["scores"][idx1]
            score2 = self.prediction_dict["scores"][idx2]
            if score1 >= score2:
                idxs_to_remove.append(idx2)
            else:
                idxs_to_remove.append(idx1)
        unique_idxs_to_remove = np.unique(idxs_to_remove)
        logging.info(f"Removing {len(unique_idxs_to_remove)} predictions")
        self._remove_instances_in_predictions(unique_idxs_to_remove)

    def _create_object_id_map(self, prediction_dict, soft_mask_threshold):
        r"Create object_id_map from predictions with the resolution of the image. 1-indexed"
        object_id_map = np.zeros(self.image_size, dtype=np.uint8)
        for idx, (box, mask) in enumerate(zip(prediction_dict["boxes"], prediction_dict["masks"])):
            y1, x1, y2, x2 = box
            mask = mask > soft_mask_threshold
            mask = self._scale_mask_to_box(mask, box)
            object_id_map[x1:x2, y1:y2][mask] = idx + 1
        return object_id_map.T
