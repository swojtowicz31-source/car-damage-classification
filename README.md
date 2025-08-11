# Car Damage Classification with CNN  
This project implements convolutional neural networks (CNN) to classify car damages into four categories: back part of the car, car windows, front part of the car, and right/left side of the car. The models are trained and evaluated using TensorFlow/Keras on a custom dataset of vehicle images.  

The dataset is included. the following folder structure inside `dataset/`:  
`dataset/train/`, `dataset/val/`, and `dataset/test/` folders, each containing subfolders named after classes (`back_part_car`, `car_windows`, `front_part_car`, `right_left_side_car`), with images inside.  

Required Python packages are TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn. Install them with:  
```bash  
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn  
