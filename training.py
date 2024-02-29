import os  # For interacting with the operating system
import sys  # For interacting with the Python interpreter
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computing
import random  # For generating random numbers

import cv2  # For computer vision tasks
import albumentations as albu  # For image augmentation

from sklearn.model_selection import train_test_split  # For splitting data into train and test sets

import torch  # PyTorch library
import torch.nn as nn  # PyTorch's neural network module
import torch.optim as optim  # For optimization algorithms
import segmentation_models_pytorch as smp  # PyTorch segmentation models
from torch.utils.data import Dataset, DataLoader  # For creating custom datasets and data loaders
from tqdm import tqdm  # For displaying progress bars during loops

SEED = 42  # Setting the random seed for reproducibility

BASE_URL = os.path.join("../input", "airbus-ship-detection")  # Defining base URL for dataset
TRAIN_URL = os.path.join(BASE_URL, "train_v2")  # URL for training data
TEST_URL = os.path.join(BASE_URL, "test_v2")  # URL for test data

def seed_everything(seed=42):
    # Setting random seed for various libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def rle_decode_the_mask(mask_series: pd.Series, shape: tuple=None):
    """
    Decode an RLE-encoded mask from a pandas series.

    Parameters:
    - mask_series (pd.Series): Pandas series containing the mask (0/"0" if empty).
    - shape (tuple, optional): Shape of the image in case of resizing.

    Returns:
    - original_shape_mask (torch.Tensor): Decoded mask with masked pixels being white.
    """
    # Extracting the mask array from the series
    mask_array = mask_series.values
        
    # Checking if the mask is empty (no ships)
    if mask_array == 0 or mask_array == '0':
        return torch.zeros(shape)  # Returning a zero tensor
    else:
        mask_array = mask_array[0] # unpacking the mask

    # Creating a background mask based on the shape
    if shape is None:
        bg_mask = torch.zeros(768 * 768, dtype=torch.uint8)
    else:
        bg_mask = torch.zeros(shape[0] * shape[1], dtype=torch.uint8)

    # Splitting the mask array to decode RLE
    mask = mask_array.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask[0:][::2], mask[1:][::2])] # take each first value for pixel posiiton and each second - for the pixels qty after it
    starts -= 1  # Adjusting the start values to 0-indexing
    ends = starts + lengths  # Calculating the end values based on start and length


    # Create a mask with the shape of the original image
    original_shape_mask = torch.zeros(768 * 768, dtype=torch.uint8)

    # Assign 1 to the pixels based on RLE encoding
    for lo, hi in zip(starts, ends):
        original_shape_mask[lo:hi] = 1 # mark the "masked" pixels as white

    # If the shape is provided and different from the original image's shape, resize the mask
    if shape is not None:
        # Reshaping the mask to the original image dimensions (just in case)
        original_shape_mask = original_shape_mask.view(768, 768)

        # Resizing the mask to match the provided shape
        original_shape_mask = cv2.resize(original_shape_mask.numpy(), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

        # Converting the resized mask back to a PyTorch tensor
        original_shape_mask = torch.from_numpy(original_shape_mask)

    return original_shape_mask.T  # Returning the decoded mask


def encode_the_image(imagePath: str, shape: tuple=None):
    """
    Encode an image into a NumPy array with optional resizing.

    Parameters:
    - imagePath (str): The path to the image file.
    - shape (tuple, optional): The desired shape of the output image (height, width).

    Returns:
    - img (numpy.ndarray): Encoded image as a NumPy array; resized; color scheme: RGB.
    """
    # Read the image from the given path
    img = cv2.imread(imagePath)
    # Convert the color space from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image if a shape is provided
    if shape is not None:
        img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
        
    return img  # Return the encoded image

class AirbusShipDetectionDataset(Dataset):
    """
    Dataset class that represents the Airbus Ship Detection data. It is utilized be the data loader to get items for batches.

    Parameters:
    - df_in (pd.DataFrame): Input dataframe containing image information.
    - augmentations (albu.Compose): Augmentation pipeline to apply to images and masks.
    - images_dir (str): Directory containing the images.
    - shape (tuple, optional): Desired shape of the output images and masks (height, width).
    """

    def __init__(self, df_in: pd.DataFrame, augmentations: albu.Compose, images_dir: str, shape=(768, 768)) -> None:
        super().__init__()
        
        self.df = df_in
        self.shape = shape
        self.augs = augmentations
        self.images_dir = images_dir

    def __len__(self) -> int:
        """
        Returns:
        - the length of the data (rows)

        """
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Get item method to retrieve an image and its mask by index. Represents an item of the dataset.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - image (torch.Tensor): The image as a PyTorch tensor (color_channel=3, self.shape[0], self.shape[1]).
        - mask (torch.Tensor): The mask as a PyTorch tensor (color_channel=1, self.shape[0], self.shape[1]).
        """
        # Get the row corresponding to the given index from the dataframe
        df_row = self.df.iloc[idx]
            
        # Extract the image name from the row
        image_name = df_row['ImageId']
        # Construct the full image path
        full_image_path = os.path.join(self.images_dir, str(image_name))

        # Encode the image using the provided shape
        image = encode_the_image(full_image_path, self.shape)
        # Decode the mask using the provided shape
        mask = rle_decode_the_mask(pd.Series(df_row['EncodedPixels']), self.shape)

        # Apply augmentations to the image and mask
        augmented = self.augs(image=image, mask=mask.numpy()) 

        # Rearrange dimensions and convert the augmented image to PyTorch tensor
        image = torch.from_numpy(augmented["image"]).permute(2, 0, 1)
        # Convert the augmented mask to PyTorch tensor and add a channel dimension
        mask = torch.from_numpy(augmented["mask"]).float().unsqueeze(0)

        # Return the processed image and mask
        return image, mask
    

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, metric_fn, device: str = "cpu", verbose: bool = True) -> dict:  
    """
    Train the model for one epoch, accumulating weights.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - loader (DataLoader): DataLoader containing the training data.
    - criterion (nn.Module): The loss function to optimize.
    - optimizer (optim.Optimizer): The optimizer algorithm to use for training.
    - metric_fn: Function to compute additional metrics (e.g., Dice coefficient).
    - device (str, optional): The device to use for training (default is "cpu").
    - verbose (bool, optional): Whether to display training progress (default is True).

    Returns:
    - logs (dict): Dictionary containing training logs (losses and metrics).
    """
    
    model.train() # Set the model to training mode

    # Initialize lists to store losses and Dices
    losses = []
    dices = []

    # Create a progress bar for training
    with tqdm(total=len(loader), desc="training", file=sys.stdout, ncols=100, disable=not verbose) as progress:
        for x_batch, y_true in loader:  # Iterate over image_mask pairs in the current batch of the loader
            
            # Move data to the specified device (CPU or GPU):
            x_batch = x_batch.to(device)
            y_true = y_true.to(device)

            # Clear previous gradients (we want weights to accumulate, not gradients)
            optimizer.zero_grad()

            # Forward pass to compute predictions
            y_pred = model(x_batch).sigmoid() # convert into binary prediction (0 or 1; approximately)

            # Compute the loss
            loss = criterion(y_pred, y_true)

            # Perform backward propagation to compute gradients
            loss.backward()

            # Append the current loss to the losses list
            losses.append(loss.item())

            # Compute and append Dices
            dice = metric_fn(y_pred, y_true).item()  # Compute the Dice coefficient
            dices.append(dice)

            # Update the progress bar with the current loss
            progress.set_postfix_str(f"loss {losses[-1]:.4f}; dice {dices[-1]:.4f}")

            # Update the model's weights based on the optimizer's strategy
            optimizer.step()

            # Update the progress bar
            progress.update(1)

    # Create a dictionary to store training logs
    logs = {
        "losses": np.array(losses),  # Array of losses for each batch
        "dices": np.array(dices)       # Array of Dices for each batch
    }
    return logs



@torch.inference_mode() # do not accumulate weights, do not evaluate gradients; the data acts as unseen for performance assesing only
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, metric_fn, device: str = "cpu", verbose: bool = True,) -> dict:
    """
    Evaluate the model on the validation set. No weights accumulation, only inference. The data doesn't take part in training.

    Parameters:
    - model (nn.Module): The neural network model to evaluate.
    - loader (DataLoader): DataLoader containing the validation data.
    - criterion (nn.Module): The loss function to use for evaluation.
    - metric_fn: Function to compute additional metrics (Dice).
    - device (str, optional): The device to use for evaluation (default is "cpu").
    - verbose (bool, optional): Whether to display evaluation progress (default is True).

    Returns:
    - logs (dict): Dictionary containing evaluation logs (losses and metrics).
    """
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store losses and dices
    losses = []
    dices = []

    with torch.no_grad(): # no gradient calculations
        # Iterate over image_mask pairs in the current batch of the loader
        for x_batch, y_true in tqdm(loader, desc="evaluation", file=sys.stdout, ncols=100, disable=not verbose):
            # Move data to the specified device (CPU or GPU)
            x_batch = x_batch.to(device)
            y_true = y_true.to(device)

            # Forward pass to compute predictions
            y_pred = model(x_batch).sigmoid()

            # Compute the loss
            loss = criterion(y_pred, y_true)

            # Append the current loss to the losses list
            losses.append(loss.item())

            # Compute and append dices
            dices.append(metric_fn(y_pred, y_true).item())  # Append the Dice value

    # Create a dictionary to store evaluation logs
    logs = {
        "losses": np.array(losses), # Array of losses for each batch
        "dices": np.array(dices)  # Array of dices for each batch
    }
    return logs


class ShipStratifiedSampler(torch.utils.data.Sampler):
    """
    A custom sampler class for stratified sampling of ship images. It stratifies images in terms of ships quantity.
    It is utilized by DataLoader and acts as a custom sampling method. If we didn't use the stratification, it could be omitted.

    Parameters:
    - data_source (pd.DataFrame): The dataset to sample from.
    - batch_size (int): The batch size.
    - multiple_ships_ratio_per_batch (float): The desired ratio of multiple (>1) ship images per batch.
    """
    def __init__(self, data_source: pd.DataFrame, batch_size: int, multiple_ships_ratio_per_batch: float):
        self.data_source = data_source # df to sample images from
        self.batch_size = batch_size # batch size
        self.multiple_ships_ratio_per_batch = multiple_ships_ratio_per_batch # the ratio of > 1 ship images per 1 batch

        self.multiple_ships_indices = data_source[data_source['Ships'] > 1].index.tolist() # multiple-ship images
        self.single_ship_indices = data_source[data_source['Ships'] == 1].index.tolist() # single-ship images
        
        # Calculate the number of batches
        self.num_batches = int(len(self.single_ship_indices) // (batch_size * (1 - self.multiple_ships_ratio_per_batch)))

        
    def __iter__(self):
        """
        Iterator function to yield batch indices.
        """
        random.seed(SEED) # ensure reproducibility
        random.shuffle(self.multiple_ships_indices) # shuffle the indices of multiple ships
        random.shuffle(self.single_ship_indices) # shuffle the indices of single ships
        
        for _ in range(self.num_batches):
            # Sample indices for multiple ships
            num_mult_ships = int(self.batch_size * self.multiple_ships_ratio_per_batch)
            sampled_mult_ships = random.sample(self.multiple_ships_indices, num_mult_ships)

            # Sample indices for non-zero ships images
            num_single_ships = self.batch_size - num_mult_ships
            sampled_single_ships = random.sample(self.single_ship_indices, num_single_ships)

            # Combine sampled indices for both types of batches
            sampled_indices = sampled_mult_ships + sampled_single_ships

            # Yield individual indices for the batch
            for idx in sampled_indices:
                yield idx

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.num_batches * self.batch_size
    

class TrainAndTestModel:
    def __init__(self, model: nn.Module, device: str, dataset_train: pd.DataFrame, dataset_val: pd.DataFrame, train_augs:albu.Compose = None, valid_augs:albu.Compose = None, images_dir: str = TRAIN_URL, input_size: tuple = (768, 768)):
        """
        Initializes a class for training and testing a deep learning model.

        Args:
            model (nn.Module): The neural network model to train and test.
            device (str): Device to run training and evaluation (e.g., "cpu" or "cuda").
            dataset_train (pd.DataFrame): Training dataset.
            dataset_val (pd.DataFrame): Validation dataset.
            train_augs (albu.Compose): Data augmentations for training. If None, resizing and normalization applied.
            valid_augs (albu.Compose): Data augmentations for validation. If None, resizing and normalization applied.
            input_size (tuple): Input image shape.
        """
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

        self.model = model
        self.device = device

        # declare arrays for storing losses and accuracies for training
        self.train_losses = []
        self.train_dices = []

        # declare arrays for storing losses and accuracies for evaluation
        self.valid_losses = []
        self.valid_dices = []

        self.batch = BATCH_SIZE
        self.workers = os.cpu_count() // 2

        self.input_size = input_size
        self.images_dir = images_dir
        self.multiple_ships_ratio_per_batch = 0.3629 # new ratio


        if train_augs is None:
            # Define a set of data augmentations for training
            train_augs = albu.Compose([
                albu.Resize(*self.input_size),
                albu.Normalize(),
            ])

        if valid_augs is None:
            # Define a set of data augmentations for validation
            valid_augs = albu.Compose([
              albu.Resize(*self.input_size),
              albu.Normalize(),
        ])


        self.train_augs = train_augs
        self.valid_augs = valid_augs


    def get_accuracies_n_losses(self):
        """
        Get the training and validation losses along with training and validation accuracies.

        Returns:
          tuple: A tuple containing four lists:
              - Training losses.
              - Validation losses.
              - Training accuracies.
              - Validation accuracies.
        """
        return self.train_losses, self.valid_losses, self.train_dices, self.valid_dices
    
    def set_multiple_ships_ratio_per_batch(self, val: float):
        """
        sets the ratio of multiple-ship images per batch during sampling
        """
        self.multiple_ships_ratio_per_batch = val


    def train(self, n_batch: int, n_epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module, metric_fn, scheduler=None): # train the model
        """
        Trains the neural network model until the specified amount of epoch.

        Args:
          n_batch (int): Batch size for training.
          n_epochs (int): Number of training epochs.
          optimizer (torch.optim.Optimizer): Optimizer for model parameters.
          loss_fn (nn.Module): Loss function to calculate training loss.
          metric_fn: Metric to consider (Dice).
          sceduler (optional): learning rate scheduler.

        Returns:
          tuple: A tuple containing four lists:
              - Training losses.
              - Validation losses.
              - Training accuracies.
              - Validation accuracies.
        """
        # Set batch size and number of workers
        self.batch = n_batch
        self.workers = os.cpu_count() // 2

        # defining the training loader
        _train_loader = DataLoader(
            AirbusShipDetectionDataset(df_in=self.dataset_train, augmentations=self.train_augs, images_dir=self.images_dir),
            batch_size=self.batch,
            sampler=ShipStratifiedSampler(data_source=self.dataset_train, batch_size=self.batch, multiple_ships_ratio_per_batch = self.multiple_ships_ratio_per_batch), # updated sampler
            num_workers=self.workers,
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
            pin_memory=False
        )

        # defining the validation loader
        _valid_loader = DataLoader(
            AirbusShipDetectionDataset(df_in=self.dataset_val, augmentations=self.valid_augs, images_dir=TRAIN_URL),
            batch_size=BATCH_SIZE,
            sampler=ShipStratifiedSampler(data_source=self.dataset_val, batch_size=self.batch, multiple_ships_ratio_per_batch = self.multiple_ships_ratio_per_batch), # updated sampler
            num_workers=self.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False
        )

        # for each epoch
        for ep in range(n_epochs):
            # print the progress
            print(f"\nEpoch {ep + 1:2d}/{n_epochs:2d}")

            # Train the model for one epoch and collect train Dice scores
            _train_logs = train_one_epoch(model=self.model, loader=_train_loader, criterion=loss_fn, optimizer=optimizer, metric_fn=metric_fn, device=self.device, verbose=True)
            _train_dice = np.mean(_train_logs["dices"]) # shown dice per epoch is the mean of all samples' dices (train)
            self.train_dices.append(_train_dice) # append to all other epochs' dices (train)
            _train_loss = np.mean(_train_logs["losses"]) # shown loss per epoch is the mean of all samples' losses (train)
            self.train_losses.append(_train_loss) # append to all other epochs' losses (train)

            print("   Training Dice:", self.train_dices[-1]) # show the last dice (train)
            print("   Training Loss:", self.train_losses[-1]) # show the last loss (train)

            # Evaluate the model on the validation dataset and collect validation Dice scores
            _valid_logs = evaluate(model=self.model, loader=_valid_loader, criterion=loss_fn, metric_fn=metric_fn, device=self.device, verbose=True)
            _valid_dice = np.mean(_valid_logs["dices"]) # shown dice per epoch is the mean of all samples' dices (evaluation)
            self.valid_dices.append(_valid_dice) # append to all other epochs' dices (evaluation)
            _valid_loss = np.mean(_valid_logs["losses"]) # shown loss per epoch is the mean of all samples' losses (evaluation)
            self.valid_losses.append(_valid_loss) # append to all other epochs' losses (evaluation)

            print("   Validation Dice:", self.valid_dices[-1]) # show the last dice (evaluation)
            print("   Validation Loss:", self.valid_losses[-1]) # show the last loss (evaluation)

        # Update the learning rate using the scheduler based on the validation metric (Dice)
        if scheduler != None:
            scheduler.step(_valid_dice)

        return self.get_accuracies_n_losses()
    
    def continue_training(self, current_epoch: int, n_batch: int, n_epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module, metric_fn, scheduler=None):
        """
        Continues training from the specified epoch. It is possible due to the tracking of all model's data within this class.

        Args:
          current_epoch (int): the last finished epoch (number).
          n_batch (int): Batch size for training.
          n_epochs (int): Number of training epochs to do.
          optimizer (torch.optim.Optimizer): Optimizer for model parameters.
          loss_fn (nn.Module): Loss function to calculate training loss.
          metric_fn: Metric to consider (Dice).
          sceduler (optional): learning rate scheduler.

        Returns:
          tuple: A tuple containing four lists:
              - Training losses.
              - Validation losses.
              - Training accuracies.
              - Validation accuracies.
        """
        self.batch = n_batch

        # defining the training loader (described in detailes above)
        _train_loader = DataLoader(
            AirbusShipDetectionDataset(df_in=self.dataset_train, augmentations=self.train_augs, images_dir=self.images_dir),
            batch_size=self.batch,
            sampler=ShipStratifiedSampler(data_source=self.dataset_train, batch_size=self.batch, zero_ships_ratio_per_batch = self.zero_ships_ratio_per_batch),
            num_workers=self.workers,
            shuffle=False,
            drop_last=True,
            persistent_workers=True,
            pin_memory=False
        )

        # defining the validation loader (described in detailes above)
        _valid_loader = DataLoader(
            AirbusShipDetectionDataset(df_in=self.dataset_val, augmentations=self.valid_augs, images_dir=TRAIN_URL),
            batch_size=BATCH_SIZE,
            sampler=ShipStratifiedSampler(data_source=self.dataset_val, batch_size=self.batch, zero_ships_ratio_per_batch = self.zero_ships_ratio_per_batch),
            num_workers=self.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False
        )

        # Continue training for additional epochs
        for ep in range(current_epoch, current_epoch + n_epochs):
            # display the progress starting with the previous epoch
            print(f"\nEpoch {ep + 1:2d}/{current_epoch + n_epochs:2d}")

            # Train the model for one epoch and collect train Dice scores
            _train_logs = train_one_epoch(model=self.model, loader=_train_loader, criterion=loss_fn, optimizer=optimizer, metric_fn = metric_fn, device=self.device, verbose=True)
            _train_dice = np.mean(_train_logs["dices"]) # shown dice per epoch is the mean of all samples' dices (train)
            self.train_dices.append(_train_dice) # append to all other epochs' dices (train)
            _train_loss = np.mean(_train_logs["losses"]) # shown loss per epoch is the mean of all samples' losses (train)
            self.train_losses.append(_train_loss) # append to all other epochs' losses (evaluation)

            print("   Training Dice:", self.train_dices[-1]) # show the last dice (train)
            print("   Training Loss:", self.train_losses[-1]) # show the last loss (train)

            # Evaluate the model on the validation dataset and collect validation IoU scores
            _valid_logs = evaluate(model=self.model, loader=_valid_loader, criterion=loss_fn, metric_fn=metric_fn, device=self.device, verbose=True)
            _valid_dice = np.mean(_valid_logs["dices"]) # shown dice per epoch is the mean of all samples' dices (evaluation)
            self.valid_dices.append(_valid_dice) # append to all other epochs' dices (evaluation)
            _valid_loss = np.mean(_valid_logs["losses"]) # shown loss per epoch is the mean of all samples' losses (evaluation)
            self.valid_losses.append(_valid_loss) # append to all other epochs' losses (evaluation)

            print("   Validation Dice:", self.valid_dices[-1]) # show the last dice (evaluation)
            print("   Validation Loss:", self.valid_losses[-1]) # show the last loss (evaluation)

        # Update the learning rate using the scheduler based on the validation metric (Dice)
        if scheduler != None:
            scheduler.step(_valid_dice)

        return self.get_accuracies_n_losses()

    def save_model(self, save_path: str):
        """
        Saves the trained model's state dictionary to a file, adding no additional keys.

        Args:
          save_path (str): Path where the model will be saved.

        Returns:
        None
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def dice_score(y_true, y_pred, smooth=1):
    """
    Compute the Dice coefficient between two binary tensors.

    Parameters:
    - y_true: Ground truth binary tensor.
    - y_pred: Predicted binary tensor.
    - smooth (float): Smoothing factor to prevent division by zero (default is 1).

    Returns:
    - dice (torch.Tensor): Dice coefficient.
    """
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


if __name__ == "__main__":
    # ensure reproducability across the script
    seed_everything()

    # import data after the EDA
    df = pd.read_csv("train_ship_segmentation_post_EDA.csv")

    # Splitting the dataframe into training and validation sets, stratifying on "Ships" column
    train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['Ships'])

    # Resetting the index of the training and validation dataframes (needed for a convenient sampling; specifically after the stratification)
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)

    # select the device to process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Going to use '{device}' device!")

    # ensure torch reproducability
    torch.manual_seed(SEED)

    # define the model architecture
    unet_model = smp.Unet("resnet18", classes=1, encoder_weights="imagenet")
    # move the model to the device
    unet_model = unet_model.to(device)
    print("Number of trainable parameters -", sum(p.numel() for p in unet_model.parameters() if p.requires_grad))

    # first milestone
    BATCH_SIZE = 16                         # batch size
    LR = 1e-4                               # learning rate
    N_EPOCHS = 5                            # number of epochs
    INPUT_SIZE = (256,256)                  # image size
    SCHEDULER = None                        # learning rate scheduler
    multiple_ships_ratio_per_batch = 0.3629 # ratio of > 1 ship-image per batch

    # train augmentations
    TRAIN_AUGS = albu.Compose([
                    albu.Resize(*INPUT_SIZE),
                    albu.Normalize(),
                ])

    # validation augmentations
    VALID_AUGS = albu.Compose([
                    albu.Resize(*INPUT_SIZE),
                    albu.Normalize(),
                ])

    # set optimizer, loss function and metric to maximize
    optimizer = optim.Adam(unet_model.parameters(), lr=LR)
    loss_fn = smp.losses.FocalLoss(mode="binary").to(device)
    metric_fn = dice_score

    unet_model_train = TrainAndTestModel(model=unet_model, device=device, dataset_train=train_df, dataset_val=validation_df, 
                                        train_augs=TRAIN_AUGS, valid_augs=VALID_AUGS, images_dir=TRAIN_URL, input_size=INPUT_SIZE)

    # set multiple ships per batch amount (stratification)
    unet_model_train.set_multiple_ships_ratio_per_batch(multiple_ships_ratio_per_batch)

    # train for N_EPOCHS
    unet_model_train.train(n_batch=BATCH_SIZE, n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn=loss_fn, metric_fn=metric_fn, scheduler=SCHEDULER)

    # second milestone
    torch.manual_seed(SEED)

    LR = 1e-5 # lower the learning rate

    # update LR
    optimizer = optim.Adam(unet_model.parameters(), lr=LR)

    # scheduler added
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=1, verbose=True)

    # continue the training from the previous state
    unet_model_train.continue_training(current_epoch=5, n_batch=BATCH_SIZE, n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn=loss_fn, metric_fn=metric_fn, scheduler=SCHEDULER)

    # third milestone
    torch.manual_seed(SEED)

    # continue the training from the previous state
    unet_model_train.continue_training(current_epoch=10, n_batch=BATCH_SIZE, n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn=loss_fn, metric_fn=metric_fn, scheduler=SCHEDULER)

    # fourth milestone
    torch.manual_seed(SEED)

    LR = 1e-10 # lower the learning rate
    N_EPOCHS=5

    # update LR
    optimizer = optim.Adam(unet_model.parameters(), lr=LR)

    # continue the training from the previous state
    unet_model_train.continue_training(current_epoch=15, n_batch=BATCH_SIZE, n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn=loss_fn, metric_fn=metric_fn, scheduler=SCHEDULER)

    # save the model
    unet_model_train.save_model(os.path.join("model", "model_2_sratified_unet_semantic_segmentation.pt"))
    print("The training is finished")