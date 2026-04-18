# deep-neural-engineering-assignment2

train.py uses the following files and directories to learn the network:

Dotted images go into directory: data/train_dots

Connected images go into: data/train_full

Each dotted image should have a connected counterpart in the other directory, using the same name.

Training uses default parameters that can be adjusted: 

LEARNING_RATE = 1e-4  #set fixed for all trainings, could be set larger when more data is used, or smaller wehn less data is used

BATCH_SIZE = 8

NUM_EPOCHS = 15   # as a max, should converge faster, following the loss, and avoid overtraining, no automatic early stopping foreseen

SAVE_DIR = "checkpoints"

Run training via:
python train.py

The script will save the model after each iteration in the checkpoints directory. 

Running test via infer_list.py
Adjust parameters if needed:

python infer_list.py --input ./data/test_input --ouput ./output/test_epoch3/ --model checkpoints/test_dataset3/model_epoch_3_dice.pt 

input: dir containing the dotted images used for testing

output: where the results are saved

model: the model file after training to be used.

