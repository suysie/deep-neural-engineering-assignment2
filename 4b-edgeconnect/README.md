# EdgeConnect preprocessing
This folder contains the `batch_edgeConnect` Jupyter notebook, which runs the pretrained EdgeConnect model from `checkpoints/` as a preprocessing step.

It processes every image in `workspace/data/test_input/` and saves the generated outputs to `workspace/3a-edgeconnect/output/`.

The resulting preprocessed images can then be used for the subsequent OCR step.