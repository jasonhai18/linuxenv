# 1 Set Up Environment
Requirements：
```
python 3.6+
tensorflow 2.0
cuda 10.0+
cudnn 2.6.0+
numpy 1.16.0+ 
opencv-python 4.1.2.30 
tqdm
EasyDict
```
## 1.1 Nvidia Driver
https://github.com/xzhengethz/linuxenv/blob/master/NvidiaDriver.md
## 1.2 Set Up Virtual Environment
### 1.3 Using pip
https://github.com/xzhengethz/linuxenv/blob/master/VirtualEnv.md
### 1.4 Using conda
#### 1.4.1 Install Anaconda
download Anaconda3-2019.10-Linux-x86_64.sh from https://www.anaconda.com/distribution/
```
cd ./
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh
```
#### 1.4.2 Create Virtual Environment with Anaconda
```
conda create -n <VirtualEnv> python=3.6 pip tensorflow <package1> <package2> ...
conda activate <VirtualEnv>
pip install opencv-python
```
# 2 Make Your Own Dataset
## 2.1 Label the Image
github repository: https://github.com/tzutalin/labelImg
```
pip install labelImg
```
Here we use the Pascal VOC format to label the image(.xml).
## 2.2 Folder
All folders are made using the Pascal VOC template.

*architecture*
```
--- data --- dataset_a --- dataset_a + year --- Annotations
	 |				    |- ImageSets --- Main
	 |				    |- JPEGImages
	 |-- dataset_b --- dataset_a + year --- Annotations
	 |				    |- ImageSets --- Main
	 |				    |- JPEGImages
         |-- dataset_c --- dataset_a + year --- Annotations
	 |				    |- ImageSets --- Main
	 |				    |- JPEGImages
```
## 2.3 Train and Validation Set
Randomly split all data into train, val and test set.
- Choose the ratio of train, val and test, normally 8:1:1.
- Modify *./Toolkit/DatasetSplit.py*
```
vim ./Toolkit/DatasetSplit.py
```
``` python
	self.anno_path = 'YourOwnDatasetPath/Annotations/'
	self.img_path = 'YourOwnDatasetPath/JPEGImages/'
	self.save_dir = 'YourOwnDatasetPath/ImageSets/Main/'
```
change the ratio
```
DatasetSplit.split_trainval(0.95, 0.05, namelist=namelist)
```
## 2.4 Convert the Pascal VOC format to the one which is used in YOLO v3 project
- Modify ./Toolkit/PVOCtoYolov3tf.py
```
vim ./Toolkit/PVOCtoYolov3tf.py
```
``` python
	self.root = 'YourOwnDatasetRootPath'
        self.image_sets = [('YourDatasetName', 'year', 'val'),('YourDatasetName', 'year', 'train'),('YourDatasetName', 'year', 'test')]
```
- Then put the *.txt* files to *./data/dataset/*
# 3 Train your model
## 3.1 Model Introduction
### 3.1.1 ./core/backbone.py 
the backbone of Yolo v3, darknet-53, return three feature maps

| Network Architecture | Algorithmn |
|---|---|
|<img width="150%" src="https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/1551aa4734added3ad0c6979ed2ed74894cdd504/docs/images/darknet53.png" style="max-width:150%;">|<img width="80%" src="https://user-images.githubusercontent.com/30433053/62342173-7ba89880-b518-11e9-8878-f1c38466eb39.png" style="max-width:70%;">|


## 3.1 Before Training
- Make new file ./data/classes/YourData.names
- Modify ./core/config.py
## 3.2 Training
```
python3 train_abb.py
```
## 3.3 Testing

# 4 Tips and Tricks
## 4.1 Learning Rate Strategy
``` python
if global_steps < warmup_steps:
	lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
else:
	lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
	    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
	)
```
<p align="center">
    <img width="30%" src="https://github.com/xzhengethz/linuxenv/blob/master/Images/LearningRateStrategy.png" style="max-width:80%;">
    </a>
</p>

## 4.2 Training Process
<p align="center">
    <img width="90%" src="https://github.com/xzhengethz/linuxenv/blob/master/Images/TensorboardLr.png" style="max-width:80%;">
    </a>
</p>

<p align="center">
    <img width="90%" src="https://github.com/xzhengethz/linuxenv/blob/master/Images/TensorboardLoss.png" style="max-width:80%;">
    </a>
</p>

