# Visual Latent Actions

> Learning Visually Guided Latent Actions for Assistive Teleoperation   
> Siddharth Karamcheti, Albert Zhai, Dylan Losey, Dorsa Sadigh  
> *Learning for Dynamics and Control (L4DC), June 2021*

Code for collecting demonstrations (on a Franka Emika Panda Robot Arm) and visual data, pretraining
visual encoders, and training visually-guided latent action controllers.

---

## Quickstart

Clones `vla` to the current working directory, then walks through dependency setup, mostly leveraging the 
`environment-<arch>` files. Assumes `conda` is installed locally (and is on your path!). Follow directions here to
install `conda` (Anaconda or Miniconda) if not: 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html.

We provide two installation directions -- one for CPU-only (e.g., MacOS, Linux) development, and one for CUDA-equipped
Linux w/ GPUs.

### Linux w/ GPU & CUDA 11.0

## Usage

### Pretraining a YOLO-v5 Detector

--- 

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary to use this code, but I 
like to keep this in the README in case things break in the future). Generally, if you're just trying to 
run/use this code, look at the Quickstart section above.

### Linux w/ GPU & CUDA 11.0

```bash
# Clone the YoloV5 Code from https://github.com/ultralytics/yolov5/tree/v2.0 (Note Version == 2.0)
git submodule add https://github.com/ultralytics/yolov5.git -b v2.0
```

---

### Note

This repository was originally written with older versions of PyTorch and PyTorch-Lightning on now deprecated version
of CUDA. If you run into any problems with the codebase -- please submit an issue, and you can expect a response within
24 hours! If urgent, please email skaramcheti@cs.stanford.edu with "VLA Codebase" in the subject line!
