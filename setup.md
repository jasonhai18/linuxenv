# Set Up Environment
Requirements：
```
python 3.6
tensorflow 2.0
numpy
opencv
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
conda create -n <VirtualEnv> python=3.6 pip tensorflow <package>
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
## Train and Validation Set
Randomly split all data into train, val and test set.
- Choose the ratio of train, val and test, normally 8:1:1.
- Modify ./Toolkit
## Convert the Pascal VOC format to the one which is used in YOLO v3 project
- Modify .py
```
```
-
-



