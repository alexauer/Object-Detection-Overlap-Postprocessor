# Overlap Processor
Process overlapping maks in object detection based instance segmentation.


## Install
Please use [Poetry](https://python-poetry.org/docs/) as virtual environment package manager. 

```bash
poetry install
```

## Usage
```bash
poetry run python main.py -f <path to the folder containing the image and the model output>
```
Attention: Please move the image and the model output file in the same folder.

## Jupyter Notebook
The notebook `data_exploration.ipynb` contains examples of the postprocessing steps.
