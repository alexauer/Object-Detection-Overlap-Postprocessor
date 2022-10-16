import argparse
import logging
from pathlib import Path

from services.instance_segmentation.engine import (
    ObjectDetectionBasedInstanceSegmentationEngine,
)

logging.basicConfig(level=logging.INFO)


def run_engine(file_path: str, export: bool = True):
    engine = ObjectDetectionBasedInstanceSegmentationEngine()
    engine.load_input_image(Path(file_path))
    _ = engine.predict()
    _ = engine.post_process()
    if export:
        engine.export_object_id_map()


def main():
    parser = init_argparse()
    args = parser.parse_args()
    run_engine(args.file_path, args.export)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, required=True)
    parser.add_argument("--export", type=bool, default=True)
    return parser


if __name__ == "__main__":
    main()
