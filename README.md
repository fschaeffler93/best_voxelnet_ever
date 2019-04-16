# Introduction
This implementation of VoxelNet is based on the work of @jeasinema and @qianguih.
It uses the TensorFlow framework and the KITTI dataset.
Several bug fixes make it run on both Windows and Linux systems.
The code is reworked for better functionality: 
- it is not longer possible to show image visualizations while training
- usage of the KITTI testing dataset is now possible without any labels

# Dependencies
- `python3` (tested on 3.6.8)
- `tensorflow`
- `numpy`
- `opencv`
- `shapely`
- `numba`
- `easydict`

# KITTI Data
Download the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
You need the following data:
- Velodyne point clouds
- Left color images of object data set
- Training labels of object data set
- Camera calibration matrices of object data set (not needed for training and validation data)

# File Structure
Split the KITTI dataset for training into **training** and **validation**.
```
└── data    <-- KITTI data directory 
    └── object 
        ├── training       <-- training data
        |   ├── image_2   
        |   ├── label_2   
        |   └── velodyne  
        ├── validation     <-- validation data
        |   ├── image_2   
        |   ├── label_2   
        |   └── velodyne 
        └── testing        <-- testing data
            ├── image_2   
            ├── calib   
            └── velodyne
```
`config.py` must be updated if the **data** folder is not located in root.

# Compilation
Compile `box_overlaps.pyx` with running `setup.py`.
```
python setup.py build_ext --inplace
```

# Usage
0. Update `config.py` if you want to use multiple GPUs.
1. Run `train.py`
```
python train.py
```
2. Run `test.py`
```
python test.py
```
See the default parameter settings in `train.py` and `test.py`. Set parameters in command line if other settings are needed.
```
python train.py --lr 0.01

python test.py --vis False
```
Predict single sample by running `test_single.py`. Set data tag with command line parameter `-t` or `--data-tag` (default `000000`).
```
python test_single.py -t 000001
```
