## IM1102 Deep Neural Engineering - assignment 2

This repository contains the code as described in the report 'Recognition of Gun-shot text with EdgeConnect and U-net preprocessing', as part of the course IM1102 of the Open University.

Directories '1..' - '6..' represent the steps of the pipeline.  
Directory 'data' contains all images (input, target, and generated).

## More specific docs
- [2a-edgeconnect-train](4b-edgeconnect-train/README.md) - EdgeConnect training setup
- [3a-edgeconnect](4b-edgeconnect/README.md) - EdgeConnect preprocessing (inference)
- [3b-unet](4c-unet/README.md) - U-Net model
- [4-ocr](5-ocr/ocr.ipynb) - OCR step
- [5-postprocessing](6-postprocessing/data_processing.ipynb) - Post-processing
- [data](data/README.md) - Dataset information

## Environment
This project uses **VS Code devcontainers** to manage runtime environments. Devcontainers allow you to define isolated Docker-based environments with specific OS versions, Python versions, and dependencies—all without affecting your system. Each step of the pipeline can have its own optimized environment.

**Devcontainers in this project:**

1. **train-edge-connect** (for training EdgeConnect model)
   - Base: NVIDIA CUDA 11.1.1 with cuDNN 8, Ubuntu 20.04
   - Python 3.7, Miniconda
   - PyTorch 1.9.1, torchvision 0.10.1
   - Used in `2a-edgeconnect-train/`

2. **use-edge-connect** (for inference with pretrained models)
   - Base: Python 3.12-slim
   - Used in `3a-edgeconnect/` for batch processing

See [VS Code devcontainer documentation](https://code.visualstudio.com/docs/devcontainers/containers) for setup instructions.


## Disclaimers
Future improvements could optimize the pipeline further. Currently, results must be manually transferred between steps—for example, outputs from one stage serve as inputs to the next. This is a limitation of the development process; a fully automated, end-to-end pipeline is planned for future iterations.
