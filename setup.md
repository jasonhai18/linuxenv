# Set Up Environment
Requirements：
```
python 3.6+
tensorflow 2.0
cuda 10.0+
cudnn 2.7.4+
numpy 1.17.4 
opencv-python 4.1.2.30 
tqdm
EasyDict
```
## Nvidia Driver
https://github.com/xzhengethz/linuxenv/blob/master/NvidiaDriver.md
## Set Up Virtual Environment
### pip
https://github.com/xzhengethz/linuxenv/blob/master/VirtualEnv.md
### conda
#### Install Anaconda
download Anaconda3-2019.10-Linux-x86_64.sh from https://www.anaconda.com/distribution/
```
cd ./
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh
```
#### Create Virtual Environment with Anaconda
```
conda create -n <VirtualEnv> python=3.6 pip tensorflow <package1> <package2> ...
conda activate <VirtualEnv>
pip install opencv-python
```
# Make Your Own Dataset
## Label the Image
github repository: https://github.com/tzutalin/labelImg
```
pip install labelImg
```
Here we use the Pascal VOC format to label the image(.xml).
## Folder
All folders are made using the Pascal VOC template.

*architecture*
```
--- data --- dataset_a --- dataset_a + year --- Annotations
	 |				    |- ImageSets --- Main
	 |				    |- jpegImages
	 |-- dataset_b --- dataset_a + year --- Annotations
	 |				    |- ImageSets --- Main
	 |				    |- jpegImages
         |-- dataset_c --- dataset_a + year --- Annotations
	 |				    |- ImageSets --- Main
	 |				    |- jpegImages
```
## Train and Validation Set
Randomly split all data into train, val and test set.
- Choose the ratio of train, val and test, normally 8:1:1.
- Modify ./Toolkit/DatasetSplit.py
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
## Convert the Pascal VOC format to the one which is used in YOLO v3 project
- Modify ./Toolkit/PVOCtoYolov3tf.py
```
vim ./Toolkit/PVOCtoYolov3tf.py
```
``` python
	self.root = 'YourOwnDatasetRootPath'
        self.image_sets = [('YourDatasetName', 'year', 'val'),('YourDatasetName', 'year', 'train'),('YourDatasetName', 'year', 'test')]
```
- Then put the .txt files to ./data/dataset/
# Train your model
## Before Training
- Make new file ./data/classes/YourData.names
- Modify ./core/config.py
## Training
```
python3 train_abb.py
```
## Testing


