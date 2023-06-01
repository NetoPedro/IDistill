# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# PyTorch Imports
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torch.utils.tensorboard import SummaryWriter

# Weights and Biases (W&B) Imports
import wandb

# Log in to W&B Account
wandb.login()

# Fix Random Seeds
random_seed = 420
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from autoencoder_utils import UNetAutoencoder
from data_utils import BonaFideImages



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, action="append", required=True, help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["BonaFideImages"], help="Data set: BonaFideImages.")

# Model
parser.add_argument('--model', type=str, required=True, choices=["UNetAutoencoder"], help='Autoencoder Name: UNetAutoencoder.')

# Embedding size
parser.add_argument('--emb_size', type=int, required=True, help="Size of the embedding.")

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Number of epochs
parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")

# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")

# Parse the arguments
args = parser.parse_args()

# Resume training
if args.resume:
    assert args.checkpoint is not None, "Please specify the model checkpoint when resume is True"

RESUME = args.resume

# Training checkpoint
CHECKPOINT = args.checkpoint

# Data directory
DATA_DIR = args.data_dir

# Dataset
DATASET = args.dataset

# Results Directory
RESULTS_DIR = args.results_dir

# Number of workers (threads)
WORKERS = args.num_workers

# Number of training epochs
EPOCHS = args.epochs

# Learning rate
LEARNING_RATE = args.lr

# Embedding size
EMB_SIZE = args.emb_size

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Resize (data transforms)
RESIZE_OPT = args.resize

# Get model name
MODEL_NAME = args.model.lower()



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(RESULTS_DIR, DATASET.lower(), MODEL_NAME, timestamp)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# Save training parameters
with open(os.path.join(results_dir, "train_params.txt"), "w") as f:
    f.write(str(args))


# Set the W&B project
wandb.init(
    project="orthomad-v2", 
    name=timestamp,
    config={
        "model": MODEL_NAME,
        "dataset": DATASET.lower(),
    }
)



# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if RESIZE_OPT == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])



# BonaFideImages
if DATASET == "BonaFideImages":

    train_set_list = list()
    
    for data_dir in DATA_DIR:
        
        # Train set
        train_set = BonaFideImages(
            data_path=data_dir,
            transform=train_transforms
        )

        train_set_list.append(train_set)
    

    if len(train_set_list) > 1:
        train_set = ConcatDataset(train_set_list)
    else:
        train_set = train_set_list[0]



# Results and Weights
weights_dir = os.path.join(results_dir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(results_dir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard"), flush_secs=30)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ResNet50
if MODEL_NAME == "UNetAutoencoder".lower():
    model = UNetAutoencoder(embedding_size=EMB_SIZE)



# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Get model summary
try:
    model_summary = summary(model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)

except:
    model_summary = str(model)


# Write into file
with open(os.path.join(results_dir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))



# Hyper-parameters
LOSS = torch.nn.MSELoss(reduction="sum")
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Watch model using W&B
wandb.watch(model)



# Resume training from given checkpoint
if RESUME:
    checkpoint = torch.load(CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    OPTIMISER.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from {checkpoint} at epoch {init_epoch}")
else:
    init_epoch = 0


# Dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=WORKERS)


# Train model and save best weights on validation set
# Initialise min_train and loss trackers
min_train_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((EPOCHS, ))


# Go through the number of Epochs
for epoch in range(init_epoch, EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print("Training Phase")


    # Running train loss
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)


    # Put model in training mode
    model.train()

    # Iterate through dataloader
    for images_orig, images_pair in tqdm(train_loader):

        # Move data and model to GPU (or not)
        images_orig, images_pair = images_orig.to(DEVICE, non_blocking=True), images_pair.to(DEVICE, non_blocking=True)

        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad(set_to_none=True)


        # Forward pass: compute predicted outputs by passing inputs to the model
        images_recons, _ = model(images_orig)

        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(images_recons, images_pair)

        # Update batch losses
        run_train_loss += loss

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}")


    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = avg_train_loss
    
    # Save it to directory
    fname = os.path.join(history_dir, f"{MODEL_NAME}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)


    # Log to W&B
    wandb_tr_metrics = {
        "loss/train":avg_train_loss,
    }
    wandb.log(wandb_tr_metrics)

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss

        print("Saving best model based on loss...")

        # Save checkpoint
        model_path = os.path.join(weights_dir, f"{MODEL_NAME}_{DATASET.lower()}_best.pt")
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': OPTIMISER.state_dict(),
            'loss': avg_train_loss,
        }
        torch.save(save_dict, model_path)

        print(f"Successfully saved at: {model_path}")


# Finish statement
print("Finished.")