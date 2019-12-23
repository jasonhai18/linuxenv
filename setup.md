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
The backbone of Yolo v3, darknet-53, return three feature maps

| Network Architecture | Algorithmn |
|---|---|
|<img width="150%" src="https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/1551aa4734added3ad0c6979ed2ed74894cdd504/docs/images/darknet53.png" style="max-width:150%;">|<img width="80%" src="https://user-images.githubusercontent.com/30433053/62342173-7ba89880-b518-11e9-8878-f1c38466eb39.png" style="max-width:70%;">|

### 3.1.2 ./core/common.py
Define several CNN algorithmns like *BatchNormalization*, *convolutional*, *residual_block*, and *upsample*.

### 3.1.3 ./core/utils.py
``` python
def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

def crop_img(image, bboxes):

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        # delta_y, delta_x = (coor[0], coor[1]), (coor[2], coor[3])
        y1, y2, x1, x2 = coor[1], coor[3], coor[0], coor[2]
        src = image[y1:y2, x1:x2, :]
        cv2.imwrite('/home/viki/aaaa/'+int(i)+'.jpg', src)

    return
```

### 3.1.4 ./core/config.py
Some hyperparameters like learning rate, training epochs, warmup epochs, batch size, data augmentation.

#### 3.1.4 ./core/yolov3.py
<p align="center">
    <img width="90%" src="https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/1551aa4734added3ad0c6979ed2ed74894cdd504/docs/images/levio.jpeg" style="max-width:80%;">
    </a>
</p>


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

