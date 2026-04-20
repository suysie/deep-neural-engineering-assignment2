# U-Net Model

## Data Structure
Training requires paired dotted and connected images in separate directories:
- Dotted images: `data/train_dots/`
- Connected images: `data/train_full/`

Each dotted image must have a corresponding connected counterpart with the same filename.

## Training

### Configuration
Adjust these parameters in `train.py` as needed:

```python
LEARNING_RATE = 1e-4      # Fixed learning rate (may increase with more data, decrease with less)
BATCH_SIZE = 8            # Batch size for training
NUM_EPOCHS = 15           # Maximum epochs; training should converge faster; no automatic early stopping
SAVE_DIR = "checkpoints"  # Directory for saving model checkpoints
```

### Run Training
```bash
python train.py
```

The script saves the model after each iteration in the `checkpoints/` directory.

## Inference

### Run Inference
```bash
python infer_list.py --input ./data/test_input \
                     --output ./output/test_epoch3/ \
                     --model checkpoints/test_dataset3/model_epoch_3_dice.pth
```

### Parameters
- `--input`: Directory containing dotted test images
- `--output`: Directory where results are saved
- `--model`: Path to the trained model file (`.pt`)

