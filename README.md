# Visual Latent Actions

> Learning Visually Guided Latent Actions for Assistive Teleoperation   
> Siddharth Karamcheti, Albert Zhai, Dylan Losey, Dorsa Sadigh  
> *Learning for Dynamics and Control (L4DC), June 2021*

Code for collecting demonstrations (on a Franka Emika Panda Robot Arm) and visual data, pretraining
visual encoders, and training visually-guided latent action controllers.

---

## Quickstart

Clones `vla` to the current working directory, then walks through dependency setup, mostly leveraging the 
`environments/environment-<arch>` files. Assumes `conda` is installed locally (and is on your path!). Follow the 
directions here to install `conda` (Anaconda or Miniconda) if not: 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html.

We provide two installation directions -- one set of instructions for CUDA-equipped machines running Linux w/ GPUs 
(for training), and another for CPU-only machines (e.g., MacOS, Linux) geared towards development and inference (e.g.,
actually running on a robot).

### Linux w/ GPU & CUDA 11.0

Ensure that you're using the `environments/environment-gpu.yaml` file --> if PyTorch doesn't build properly for your
setup, checking the CUDA Toolkit is usually a good start (e.g., via a call to `nvcc -V`; should return 11.0). The 
current file only supports CUDA 11.0 (additional CUDA versions can be supported, please file an issue!).

```bash
# Clone `vla` Repository, *with* all submodules!
git clone --recurse-submodules https://github.com/Stanford-ILIAD/vla.git
cd vla
conda env create -f environments/environment-gpu.yaml
conda activate vla
```

### Mac OS & Linux (CPU)

Ensure that you're using the `environments/environment-cpu.yaml` file.

```bash
# Clone `vla` Repository, *with* all submodules!
git clone --recurse-submodules https://github.com/Stanford-ILIAD/vla.git
cd vla
conda env create -f environments/environment-cpu.yaml
conda activate vla
```

---

## Usage

Getting a Visually Guided Latent Actions pipeline up and running requires the following 4 steps:

1. Pretraining a YOLO-v5 Detector on an "in-domain" dataset of images of your static robot workspace, with your 
   different objects in various positions and alignments.
   
2. Collecting a series of kinesthetic demonstrations specifying the desired tasks you want for each object (recall 
   that latent action models are task-conditional).
   
3. Training a Latent Actions Model on the collected demonstrations, using the in-domain pre-trained YOLO-v5 model to 
   detect objects.
   
4. Using your visually guided latent actions model to assist a user in teleoperating the robot.

In this entire repository, we assume the robot you are using is teh Franka Emika Panda Arm, controlled via the provided
libfranka C++ controllers.

#### Pretraining a YOLO-v5 Detector

To pretrain a YOLO-v5 detector, please follow the detailed the directions in the README in the `pretrain-detector/`
directory. You will need to collect a series of images of different objects on your workspace in various positions,
annotate them (helper script provided), then use the `yolov5/` submodule to train a YOLO-v5 model from scratch. 

The README walks you through each of these steps.

#### Collecting Kinesthetic Demonstrations with the Franka Emika Panda Arm

After pretraining the YOLO-v5 detector, you need to collect a set of kinesthetic demonstrations (usually from 3 - 10) 
for each task/object in your dataset. Code for recording kinesthetic demonstrations can be found in `record.py` in
the top-level directory of this repository. You should run this script once for each task/object you care about (e.g.,
when collecting 10 demonstrations for pushing a can north, run with:

```bash
# Name should contain the object name, demonstration path should be consistent across all tasks/objects!
python record.py --name can-north --demonstration_path demos/
``` 

Note that we call custom libfranka C++ code (`readState`) to monitor robot joint positions for collecting 
demonstrations. For reproducibility, all low-level robot code we use can be found in `robot/low-level`. You will need
to follow the directions here to build this for your Robot: https://frankaemika.github.io/docs/installation_linux.html 

#### Training a YOLO + Latent Actions Model

After collecting demonstrations (assuming demos pickle files are saved in `demos/`), you can train a Visually-Guided
Latent Actions model. To do this, use `train.py` at the top-level of this repository:

```bash
# Pass in path to demonstrations, classes = [list of names of objects]
python train.py --demonstrations demos/ --n_classes 3 --classes apple banana cherry --yolo_model <path to yolo>
```

This will train a conditional auto-encoder with our default (from paper) hyperparameters; feel free to edit and change
these as you see fit. This will store a final model checkpoint in `checkpoints/` with the current code; feel free to
edit this as well.

#### Assistive Teleoperation w/ Visually Guided Latent Actions

Finally, after training a visually grounded latent actions model, you can use it to assist in teleoperation, using the
top-level script `teleoperate.py`:

```bash
# Run teleoperation WITHOUT Latent Actions (Pure End-Effector Control)
python teleoperate.py --model endeff

# Run teleoperation WITH Latent Actions (Make sure to change `CHECKPOINT` to point to the LA model checkpoint!)
python teleoperate.py --model la --yolo_model <path to yolo> --n_classes 3
```

This runs latent actions assisted teleoperation with our default parameters; please tweak and change these to better
suit your usecase, robot workspace, controllers, etc.

--- 

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary to use this code, but I 
like to keep this in the README in case things break in the future). Generally, if you're just trying to 
run/use this code, look at the Quickstart section above.

### Linux w/ GPU & CUDA 11.0

```bash
# Clone the YoloV5 Code from https://github.com/ultralytics/yolov5/tree/v2.0 (Note Version == 2.0)
git submodule add https://github.com/ultralytics/yolov5.git -b v2.0

# Create Python (Conda) Environment
conda create --name vla python=3.8
conda activate vla
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install ipython jupyter
conda install pytorch-lightning -c conda-forge

pip install black matplotlib opencv-python pygame typed-argument-parser
```

### Mac OS & Linux (CPU)

```bash
# Clone the YoloV5 Code from https://github.com/ultralytics/yolov5/tree/v2.0 (Note Version == 2.0)
git submodule add https://github.com/ultralytics/yolov5.git -b v2.0

# Create Python (Conda) Environment
conda create --name vla python=3.8
conda activate vla
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython jupyter
conda install pytorch-lightning -c conda-forge

pip install black matplotlib opencv-python pygame typed-argument-parser
```

---

### Note

This repository was originally written with older versions of PyTorch and PyTorch-Lightning on now deprecated version
of CUDA. If you run into any problems with the codebase -- please submit an issue, and you can expect a response within
24 hours! 

If urgent, please email skaramcheti@cs.stanford.edu with "VLA Code" in the subject line!
