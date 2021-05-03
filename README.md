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

#### Collecting Kinesthetic Demonstrations with the Franka Emika Panda Arm

#### Training a YOLO + Latent Actions Model

#### Assistive Teleoperation w/ Visually Guided Latent Actions

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
24 hours! If urgent, please email skaramcheti@cs.stanford.edu with "VLA Codebase" in the subject line!
