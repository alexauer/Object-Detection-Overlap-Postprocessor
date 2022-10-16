from pathlib import Path


from services.instance_segmentation.engine import ObjectDetectionBasedInstanceSegmentationEngine
from services.settings import ROOT_DIR


def test_load_input_image():
    engine = ObjectDetectionBasedInstanceSegmentationEngine()
    engine.load_input_image(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))

    assert engine.image
    assert engine.image.size == (1000, 1000)


def test_predict():
    engine = ObjectDetectionBasedInstanceSegmentationEngine()
    _ = engine.load_input_image(Path(ROOT_DIR, "test", "fixtures", "test_image.png"))
    predictions = engine.predict()

    assert type(predictions) == dict
    assert "boxes" in predictions
    assert "scores" in predictions
    assert "class_ids" in predictions
    assert "masks" in predictions
