# Pretraining the YOLO-v5 Detector

In order to pretrain a YOLO-v5 Detector using the `yolov5` submodule (from ultralytics), we need to collect and prepare
a real-world image dataset. We used the following steps to do this.

+ First, collect several images of your workspace populated with all the objects you want to detect using a fixed 
  camera (this should be the camera's location at inference time as well!). Vary the number, position, and orientation
  of your objects across the different images as well. We used Ubuntu's built-in image capture application to do this,
  which took around 15 minutes total.
  
+ Next, set up the proper file structure in this directory (`pretrain-detector`). Put your images in a folder and name 
  it `images`. Make an empty folder under the same parent directory, and name it `labels`. We have already created
  these folders and populated them with a single sample image for your reference.
  
+ Next is the annotation step. From this directory (`pretrain-detector`) run `python label.py` to drop into a labeling
  loop for each image in `images/`. Mouse over each object in an image, then press the corresponding class number (0-9)
  to record labels for an image (the console should update). Hit `u` to undo a label if you've made a mistake, and `n`
  to commit labels and move on to the next image. The script will save labels after every image -- feel free to update
  the script to better suit your workflow.
  
+ We also need to make a YAML file to summarize the dataset. Change the classes and file locations in `sample.yaml`, 
  and put it in the `data` subdirectory of the YOLOv5 repository (`../yolov5`). You can rename the file to whatever 
  you want (e.g. `vla-<date>.yaml`).
  
+ Similarly, put the edited `sample-yolo.yaml` in the `models` subdirectory of your YOLOv5 repository. Edit it with the
  number of classes `nc` you specifically have for your dataset.
  
+ Finally, we can run the training script. Change directories to the YOLOv5 base directory, and run 
  `python train.py --img 640 --batch 8 --epochs 60 --data ./data/sample.yaml --cfg ./models/sample-yolo.yaml --noautoanchor`. 
  Replace `sample.yaml` and `sample-yolo.yaml` if you renamed them. You can also specify a weight initialization with 
  the `--weights` argument (for example to use an existing pretrained YOLO model -- we did not do this in our work). 
  The batch size and number of epochs can be tuned over runs. The `--img` argument specifies the input resolution. 
  For 640, the image will be shrunk so that the width is 640 pixels, and then the top and bottom will be padded with 
  gray to reach a size of 640x640.

+ A folder will be created in `runs` that will have a `weights` folder inside. This is your trained model! For testing, 
  see `test.py` in the YOLOv5 repository. The rest of the parent repository provides code for using this pretrained 
  model in the VLA pipeline!
