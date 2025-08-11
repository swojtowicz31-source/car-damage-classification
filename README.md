# Car Damage Classification with CNN

This project implements convolutional neural networks (CNN) to classify car damages into four categories: back part of the car, car windows, front part of the car, and right/left side of the car. Models are trained and evaluated using TensorFlow/Keras on a custom vehicle image dataset.

## Dataset

The dataset is **not included** due to size limitations. Prepare or download your dataset with the following folder structure inside a `dataset/` directory:

dataset/
├── train/
│ ├── back_part_car/
│ ├── car_windows/
│ ├── front_part_car/
│ └── right_left_side_car/
├── val/
│ ├── back_part_car/
│ ├── car_windows/
│ ├── front_part_car/
│ └── right_left_side_car/
└── test/
├── back_part_car/
├── car_windows/
├── front_part_car/
└── right_left_side_car/

csharp
Kopiuj
Edytuj

Each subfolder should contain images for the respective class.

## Requirements

Install required Python packages with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
Usage
Run the training script to train the CNN on your dataset.

Evaluate the model using provided scripts.

Visualize training progress, confusion matrix, and ROC curves.

Adjust dataset paths and parameters as needed for your setup.

javascript
Kopiuj
Edytuj

