from pathlib import Path

import numpy as np
from PIL import Image

from services.instance_segmentation.processor import OverlapPostProcessor
from services.settings import ROOT_DIR


def test_remove_duplicates():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 1]]),
        "scores": np.array([0.9, 0.9, 0.8]),
        "class_ids": np.array([1, 1, 1]),
        "masks": np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    deduplicated_predictions = processor._remove_duplicates()

    assert "boxes" in deduplicated_predictions
    assert "scores" in deduplicated_predictions
    assert "class_ids" in deduplicated_predictions
    assert "masks" in deduplicated_predictions

    assert len(deduplicated_predictions["boxes"]) == 2
    assert len(deduplicated_predictions["scores"]) == 2
    assert len(deduplicated_predictions["class_ids"]) == 2
    assert len(deduplicated_predictions["masks"]) == 2

    assert np.array_equal(deduplicated_predictions["boxes"], np.array([[0, 0, 1, 1], [1, 0, 1, 1]]))
    assert np.array_equal(deduplicated_predictions["scores"], np.array([0.9, 0.8]))


def test_remove_instance_in_predictions():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 1]]),
        "scores": np.array([0.9, 0.9, 0.8]),
        "class_ids": np.array([1, 1, 1]),
        "masks": np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    processor._remove_instances_in_predictions(2)

    assert "boxes" in processor.prediction_dict
    assert "scores" in processor.prediction_dict
    assert "class_ids" in processor.prediction_dict
    assert "masks" in processor.prediction_dict

    assert len(processor.prediction_dict["boxes"]) == 2
    assert len(processor.prediction_dict["scores"]) == 2
    assert len(processor.prediction_dict["class_ids"]) == 2
    assert len(processor.prediction_dict["masks"]) == 2

    assert np.array_equal(processor.prediction_dict["boxes"], np.array([[0, 0, 1, 1], [0, 0, 1, 1]]))
    assert np.array_equal(processor.prediction_dict["scores"], np.array([0.9, 0.9]))


def test_image_size():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1]]),
        "scores": np.array([0.9]),
        "class_ids": np.array([1]),
        "masks": np.array([[[1, 1]]]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)

    assert processor.image_size == (1000, 1000)


def test_find_overlaps():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1], [10, 10, 11, 11]]),
        "scores": np.array([0.9, 0.9, 0.8]),
        "class_ids": np.array([1, 1, 1]),
        "masks": np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    overlaps = processor._find_overlaps()

    assert len(overlaps) == 2
    assert np.array_equal(overlaps[0], np.array([0, 1]))
    assert np.array_equal(overlaps[1], np.array([1, 0]))


def create_mask(box_area, soft_mask_value: float = 0.5):
    r"""Create 28x28 mask with a bounding box area where every value gets the oft_mask_value assigned.
    box_area is a bounding box with [x1, y1, x2, y2] coordinates.
    """
    mask = np.zeros((28, 28))
    mask[int(box_area[1]) : int(box_area[3]), int(box_area[0]) : int(box_area[2])] = soft_mask_value
    return mask


def test_scale_mask_to_box():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png")).resize((100, 100))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1], [10, 10, 11, 11]]),
        "scores": np.array([0.9, 0.9, 0.8]),
        "class_ids": np.array([1, 1, 1]),
        "masks": np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    }

    mock_mask = create_mask([0, 0, 5, 5], soft_mask_value=0.5)
    mock_box = np.array([0, 0, 50, 50])

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    scaled_mask = processor._scale_mask_to_box(mock_mask, mock_box)
    print(scaled_mask)
    assert scaled_mask.shape == (50, 50)
    assert scaled_mask[0, 0] == 0.5
    assert scaled_mask[5, 5] == 0.5
    assert scaled_mask[0, 10] == 0.0
    assert scaled_mask[10, 0] == 0.0


def test_create_object_id_map():
    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png")).resize((56, 56))
    mock_mask1 = create_mask([0, 0, 27, 27], 0.8)
    mock_mask2 = create_mask([0, 0, 27, 27], 0.9)
    mock_predictions = {
        "boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
        "scores": np.array([0.9, 0.9]),
        "class_ids": np.array([1, 1]),
        "masks": np.array([mock_mask1, mock_mask2]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    object_id_map = processor._create_object_id_map(mock_predictions, 0.5)

    assert object_id_map.shape == (56, 56)
    assert object_id_map[0, 0] == 1
    assert object_id_map[0, 9] == 1
    assert object_id_map[9, 0] == 1
    assert object_id_map[0, 10] == 0
    assert object_id_map[10, 0] == 0

    assert object_id_map[20, 20] == 2
    assert object_id_map[20, 29] == 2
    assert object_id_map[29, 20] == 2
    assert object_id_map[20, 30] == 0
    assert object_id_map[0, 20] == 0
    assert object_id_map[20, 0] == 0

    assert np.unique(object_id_map).shape == (3,)


def test_filter_by_confidence():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1], [10, 10, 11, 11]]),
        "scores": np.array([0.9, 0.8, 0.8]),
        "class_ids": np.array([1, 1, 1]),
        "masks": np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    overlaps = processor._find_overlaps()

    assert len(overlaps) == 2

    processor._filter_by_confidence_if_overlaps_exist(overlaps)
    filtered_predictions = processor.prediction_dict

    assert len(filtered_predictions["boxes"]) == 2
    assert len(filtered_predictions["scores"]) == 2
    assert len(filtered_predictions["class_ids"]) == 2
    assert len(filtered_predictions["masks"]) == 2


def test_distribute_by_confidence():

    mock_image = Image.open(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    mock_predictions = {
        "boxes": np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1], [10, 10, 11, 11]]),
        "scores": np.array([0.9, 0.8, 0.8]),
        "class_ids": np.array([1, 1, 1]),
        "masks": np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]),
    }

    processor = OverlapPostProcessor(mock_image, mock_predictions)
    overlaps = processor._find_overlaps()
    processor._distribute_by_confidence_if_overlaps_exist(overlaps)
    filtered_predictions = processor.prediction_dict

    assert len(filtered_predictions["boxes"]) == 3
    assert len(filtered_predictions["scores"]) == 3
    assert len(filtered_predictions["class_ids"]) == 3
    assert len(filtered_predictions["masks"]) == 3

    assert filtered_predictions["scores"][0] == 0.8
    assert filtered_predictions["scores"][1] == 0.8
    assert filtered_predictions["scores"][2] == 0.9
