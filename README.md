# ratio_exp
This project runs basic CNN ratio estimation experiments. Modules can be added to support more image resolutions and styles.

## Install
Clone this repository and run install.sh in the root directory.

## Run basic ratio estimation experiment
Run `test_ratioRegression/test_dataGenerate/test_generate_ratio_dataset.py` and `test_ratioRegression/test_train/test_trainer.py`

## Add image styles / textures
Edit `ratioExperiments/ratioReg/ClevelandMcGill/bar_figure_typeX.py`

## Training / validation / test data manipulation
Edit `_get_TrainTestVal_ratio()` method in `ratioExperiments/ratioReg/dataGenerate/GenerateRatioDataset.py`

## Adding models
Resnet, VGG and ViT are already implemented. You can write your own model and put it under `ratioExperiments/ratioReg/models`, and include it in `ratioExperiments/ratioReg/train/trainer.py`
