# EdgeConnect training
This folder contains a modified version of the EdgeConnect repository, adapted to work with our environment.

**Setup:** Use the `train-edge-connect` devcontainer. Refer to [VS Code devcontainer documentation](https://code.visualstudio.com/docs/devcontainers/containers) for setup instructions.

**Data preparation:** Run `make_masks_and_flists.py` to:
- Generate training masks by computing the difference between target and input images
- Create the flist files required for training

Training artifacts are saved in the `checkpoints/` folder.