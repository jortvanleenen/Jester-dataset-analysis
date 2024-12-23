# Analysis of the Jester Dataset

This repository contains the code and resources used in the research paper *"Comprehensive Analysis of the Jester Dataset Using State-of-the-Art Video Classification Models"*.

The paper thoroughly details our findings. Although it has not been formally published, access can be arranged by contacting one of the contributors to this repository.

## Contents

The repository includes the following:

1. **Scripts for Dataset Preprocessing**:
   - A script for calculating the mean and standard deviation of the Jester dataset frames;
   - Code used to generate training, validation, and test splits from the dataset;

2. **Dataset Splits**:
   - The generated training, validation, and test splits, provided as space-separated text files;

3. **Model Training and Testing**:
   - Code for training and testing state-of-the-art video classification models on the Jester dataset.

## Dependencies

The models in this repository are built using **PyTorchVideo**, which is partly based on and compatible with **PyTorch** and specifically **torchvision**.
In addition, we use various packages for data handling, image processing, and numerical operations, such as **Pillow**, **NumPy** and **tqdm**.

## License

This repository is licensed under the GNU General Public License v3.0 (GPLv3).
