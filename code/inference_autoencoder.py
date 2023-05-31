# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# PyTorch Imports
import torch
import torchvision



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

# Output directory
parser.add_argument("--new_data_dir", type=str, required=True, help="Directory to save the embeddings.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["BonaFideImages"], help="Data set: BonaFideImages.")

# Model
parser.add_argument('--model', type=str, required=True, choices=["UNetAutoencoder"], help='Autoencoder Name: UNetAutoencoder.')

# Embedding size
parser.add_argument('--emb_size', type=int, required=True, help='Embedding size.')

# Image size
parser.add_argument('--imgsize', type=int, default=224, help="Size of the image after transforms")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# Checkpoint
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of the model")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Parse the arguments
args = parser.parse_args()




# Data directory
DATA_DIR = args.data_dir

# Results Directory
NEW_DATA_DIR = args.new_data_dir

# Dataset
DATASET = args.dataset

# Number of workers (threads)
WORKERS = args.num_workers

# Image size (after transforms)
IMG_SIZE = args.imgsize

# Get model name
MODEL_NAME = args.model.lower()

# Embedding size
EMB_SIZE = args.emb_size

# Checkpoint
CHECKPOINT = args.checkpoint




# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE


# Train Transforms
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])



# BonaFideImages
if DATASET == "BonaFideImages":

    images_fpaths = list()
    
    for data_dir in DATA_DIR:
        
        # Train set
        train_set = BonaFideImages(
            data_path=data_dir,
            transform=transforms
        )

        train_set_imgs = train_set.images_fnames
        train_set_imgs_fpaths = [os.path.join(data_dir, img_fpath) for img_fpath in train_set_imgs]
        images_fpaths += train_set_imgs_fpaths



# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ResNet50
if MODEL_NAME == "UNetAutoencoder".lower():
    model = UNetAutoencoder(embedding_size=EMB_SIZE)



# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Load checkpoint
checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
print(f"Loading weights from {CHECKPOINT}.")



# Put model in evaluation modetrain_loader
model.eval()


# Iterate through the list of images and save it
with torch.no_grad():
    for image_fpath in tqdm(images_fpaths):

        # Open PIL image
        pil_image = Image.open(image_fpath).convert("RGB")

        # Convert to Tensor
        tensor_image = transforms(pil_image)

        # Add a batch dimension
        tensor_image = torch.unsqueeze(tensor_image, dim=0)

        # Move data and model to GPU (or not)
        tensor_image = tensor_image.to(DEVICE, non_blocking=True)

        # Forward pass: compute predicted outputs by passing inputs to the model
        _, embedding = model(tensor_image)

        # Convert this embedding to NumPy
        embedding = embedding.detach().cpu().numpy()

        # Create a path to save embedding
        embedding_fpath = os.path.join(NEW_DATA_DIR, image_fpath.split("/")[-2])
        if not os.path.isdir(embedding_fpath):
            os.makedirs(embedding_fpath)
        
        # Get the complete filename/path of the embedding
        embedding_fname = os.path.join(embedding_fpath, image_fpath.split("/")[-1])

        # Save it as NumPy array
        np.save(
            file=embedding_fname,
            arr=embedding,
            allow_pickle=True,
            fix_imports=True
        )



# Finish statement
print("Finished.")
