# Set up environment
Requirements：

python 3.6

tensorflow 2.0

numpy

opencv

tqdm

EasyDict

## Nvidia driver
https://github.com/xzhengethz/linuxenv/blob/master/NvidiaDriver.md
## Set upo
### pip
https://github.com/xzhengethz/linuxenv/blob/master/VirtualEnv.md
### conda
#### install anaconda
download Anaconda3-2019.10-Linux-x86_64.sh from https://www.anaconda.com/distribution/
```
cd ./
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh
```
#### create virtual environment with anaconda
```
conda create -n <VirtualEnv> python=3.6 pip tensorflow <package>
conda activate <VirtualEnv>
pip install opencv-python
```
# make your own dataset
## label the image
github repository: https://github.com/tzutalin/labelImg
```
pip install labelImg
```
Here we use the Pascal VOC format to label the image(.xml).
## folder
All folders are made using the Pascal VOC template.
## Convert the Pascal VOC format to the one which is used in YOLO v3 project
-
-
-



