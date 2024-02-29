import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computing
import random  # For generating random numbers

import cv2  # For computer vision tasks
import albumentations as albu  # For image augmentation

import matplotlib.pyplot as plt  # For plotting
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets

import torch  # PyTorch library
import torch.nn as nn  # PyTorch's neural network module
import segmentation_models_pytorch as smp  # PyTorch segmentation models

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


def plot_the_image(imagePath: str, axis, shape: tuple=None):
    """
    Plot an image on a given axis with optional resizing.

    Parameters:
    - imagePath (str): The path to the image file.
    - axis: The matplotlib axis to plot the image.
    - shape (tuple, optional): The desired shape of the output image (height, width).
    """
    # Check if the image exists
    if not os.path.exists(imagePath):
        print("The image", imagePath, "is not found")
        return
    
    # Encode the image
    img = encode_the_image(imagePath, shape)
    
    # Resize the image if a shape is provided
    if shape is not None:
        img = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)

    # Plot the image on the given axis
    if axis is not None:
        axis.imshow(img)
        axis.axis('off')


def plot_the_mask(mask_array: pd.Series, axis=None, shape: tuple=None):
    """
    Plot a mask on a given axis with optional resizing.

    Parameters:
    - mask_array (pd.Series): Pandas series containing the mask.
    - axis: The matplotlib axis to plot the mask.
    - shape (tuple, optional): The desired shape of the output mask (height, width).

    Returns:
    - None
    """
    # Decode the mask
    mask_decoded = rle_decode_the_mask(mask_array, shape)

    # Convert mask_decoded to numpy array
    mask_decoded_np = mask_decoded.numpy()

    # Reshape the mask if shape is provided
    if shape is not None:
        mask_decoded_resized = cv2.resize(mask_decoded_np, shape[::-1], interpolation=cv2.INTER_LINEAR)

    # Plot the mask on the given axis
    if axis is not None:
        axis.imshow(mask_decoded_resized, cmap='gray')
        axis.axis('off')


def apply_threshold(mask: np.ndarray, threshold: float):
    """
    Applies a threshold to a mask.

    Parameters:
    - mask (numpy.ndarray): Input predicted mask array.
    - threshold (float): Threshold value.

    Returns:
    - thresholded_mask (numpy.ndarray): Thresholded mask array.
    """
    return (mask > threshold).astype(np.uint8)

def get_prediction(model: nn.Module, image_name: str, shape: tuple=(256,256)):
    """
    Get predictions for a given image using the specified model.

    Parameters:
    - model (nn.Module): Trained model.
    - image_name (str): Name of the image file.
    - shape (tuple): height and width of the final mask

    Returns:
    - predicted_mask (numpy.ndarray): Predicted mask array if the image is within the validation data, None otherwise.
    """
    # Check if the image is in the validation data
    if len(validation_df[validation_df['ImageId'] == image_name]) == 0:
        print("The", image_name, "doesn't exist or may be in a training data. We need to assess the quality utilizing the unseen data only")
        return None

    # Read the image using OpenCV and convert it to RGB color space
    image = cv2.imread(os.path.join(TRAIN_URL, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the desired input size using bilinear interpolation
    image = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)

    # Apply validation augmentations to the image
    image = VALID_AUGS(image=image)['image']

    # Convert the image to a PyTorch tensor, permute dimensions, add batch dimension, and move to the device
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) # from (width, height, color_channel) to (batch_size=1, color_channel, width, height)

    # Perform prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predicted_mask = model(image_tensor).sigmoid().squeeze().cpu().numpy() # convert to probabilities, reshape into (width, height), move to cpu and cast to numpy

    return predicted_mask



if __name__ == "__main__":
    # ensure reproducability across the script (images are still sampled randomly)
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

    # define the model architecture
    unet_model = smp.Unet("resnet18", classes=1, encoder_weights="imagenet")
    # move the model to the device
    unet_model = unet_model.to(device)
    print("Number of trainable parameters -", sum(p.numel() for p in unet_model.parameters() if p.requires_grad))

    # ensure on warnings or errors if working on CPU, then load the best model weights
    if device == "cpu":
        checkpoint = torch.load(os.path.join("models", "model_2_sratified_unet_semantic_segmentation.pt"), map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join("models", "model_2_sratified_unet_semantic_segmentation.pt"))

    # load imported weights into the model
    unet_model.load_state_dict(checkpoint)

    # image size to plot and predict (model was trained on (256,256))
    INPUT_SIZE = (256,256)
    # the threshold that divides predictions into "non-ship pixel" and "ship-pixel"
    THRESHOLD = 4e-4

    # validation augmentations (same that the model was trained on)
    VALID_AUGS = albu.Compose([
                albu.Resize(*INPUT_SIZE),
                albu.Normalize(),
            ])

    # Filter images by ship count (no ships, 1 ship, multiple ships)
    no_ships_images = validation_df[validation_df['Ships'] == 0].sample(3)['ImageId'].tolist()
    single_ship_images = validation_df[validation_df['Ships'] == 1].sample(3)['ImageId'].tolist()
    multiple_ships_images = validation_df[validation_df['Ships'] > 1].sample(3)['ImageId'].tolist()

    # Determine the maximum number of rows needed for the subplot grid
    ROWS_PER_IMAGE = max(len(no_ships_images), len(single_ship_images), len(multiple_ships_images)) * 3

    # Create a figure and subplots for displaying images and masks
    fig = plt.figure(figsize=(20, 35))
    axis = fig.subplots(ROWS_PER_IMAGE, 3)

    # Function to plot images and masks
    def plot_images(image_list, ships, title_suffix):
        for i, image_id in enumerate(image_list):
            plot_the_image(os.path.join(TRAIN_URL, image_id), axis[i*3 + ships, 0], INPUT_SIZE)
            axis[i*3 + ships, 0].set_title(f'Input Image ({title_suffix})')
            axis[i*3 + ships, 0].axis('off')

            plot_the_mask(validation_df[validation_df['ImageId'] == image_id]['EncodedPixels'], axis[i*3 + ships, 1], INPUT_SIZE)
            axis[i*3 + ships, 1].set_title('Ground Truth Mask')
            axis[i*3 + ships, 1].axis('off')

            predicted_mask = get_prediction(unet_model, image_id, INPUT_SIZE)
            predicted_mask = apply_threshold(predicted_mask, THRESHOLD)

            axis[i*3 + ships, 2].imshow(predicted_mask, cmap='gray')
            axis[i*3 + ships, 2].set_title('Predicted Mask')
            axis[i*3 + ships, 2].axis('off')

    # Plot images for each category
    plot_images(no_ships_images, 0, '0 ships')
    plot_images(single_ship_images, 1, '1 ship')
    plot_images(multiple_ships_images, 2, '> 1 ship')

    plt.tight_layout()
    plt.show()