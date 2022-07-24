# CS50-3-Model

Damian Chimes - dchi4033@uni.sydney.edu.au
Siyao Jiang - sjia4014@uni.sydney.edu.au
Qingyi Li - qili5278@uni.sydney.edu.au
Chenhui Lyu - clyu7944@uni.sydney.edu.au
Jerry Wang - jwan7597@uni.sydney.edu.au
Angie Zhang - yzha4981@uni.sydney.edu.au

COVIDx Dataset - https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

Our new external dataset - https://www.kaggle.com/datasets/angiezhang01/cs503external-validation

## In order to replicate our results, please perform the following steps:
1. Clone the repository to a local folder.
2. Create “./data/” subdirectory with subfolders “external”, “train”, “test”, and “input”.
3. Download the COVIDx and external dataset with their label files into the respective
subfolder in “./data/. Links available on the GitHub page.
4. Run the three *_data_prep.py files in the order of COVIDx, external, then 10-fold, to
preprocess and convert images into NumPy files for faster load time.
5. Run the CS50-3 Model.ipynb notebook to evaluate model performances.


## Package Depencencies:
• NumPy 1.20.2
• Pandas 1.2.4
• Pillow 8.2.0
• Scikit-Learn 1.0.2
• TensorFlow 2.8.0, with GPU support
