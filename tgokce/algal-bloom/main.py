import os
import matplotlib.pyplot as plt
import rionegrodata
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, RasterDataset
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torchgeo.samplers.constants import Units

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.mkdir("results")
os.chdir("results")

# Machine Learning Model - Linear Classifier.
class LinearClassifier(torch.nn.Module):
    # Input dimension refers to the each data sample in the input image and output dimension is the number of classes.
    def __init__(self, input_dim=1, output_dim=5):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

# Pre-calculated values that are required for normalization and clipping of the data of Palmar reservoir.
_CLIP = 150
_MEAN = [2.37, 3.25, 1.30, 26.04, 17.50, 42.33, 0.75, 218.71, 74.00, -1.11, 0.14]
_STD = [0.98, 0.44, 0.46, 8.72, 5.56, 34.48, 1.28, 108.29, 13.37, 2.83, 2.79]
_TRANSFORM = [True, True, True, False, False, False, True, False, False, False, False]

# Bins used for the classification of the data.
_BINS = [0, 10, 30, 75]

# Weights of each bucket created after the pre-processed data of Palmar reservoir is assigned to the corresponding bin values.
_WEIGHTS = torch.tensor([0.0, 1/63.574, 1/23.185, 1/7.963, 1/5.273]).to(device)

# Pre-processing for combining data modalities
def input_preprocess(data):
    data = data.float()

    # Clip data to avoid extreme values.
    data = torch.clamp(data, 1e-6, _CLIP)

    # Apply Jeo-Yohnson transform to certain channels.
    for i in range(data.shape[2]):
        data[:, :, i, :, :]=torch.nan_to_num(data[:, :, i, :, :], nan=_MEAN[i])
        if _TRANSFORM[i]:
            data[:, :, i, :, :] = torch.log(data[:, :, i, :, :] + 1)

    # Normalize data.
    mean = torch.tensor(_MEAN)[None, None, :, None, None]
    std = torch.tensor(_STD)[None, None, :, None, None]
    data = (data - mean) / std

    # Permute temporal dimensions.
    data = torch.permute(data, (0, 3, 4, 1, 2))

    # Collapse temporal and band dimensions.
    data = torch.flatten(data, start_dim=0, end_dim=2)

    # Collapse temporal and band dimensions.
    data = torch.flatten(data, start_dim=1, end_dim=2)

    return data


def modality_preprocess(data, index):
    data = data.float()

    # Clip data to avoid extreme values.
    data = torch.clamp(data, 1e-6, _CLIP)

    # Apply Yeo-Johnson transform to certain channels.
    for i in range(data.shape[2]):
        data[:, :, i, :, :]=torch.nan_to_num(data[:, :, i, :, :], nan=_MEAN[i])
        if _TRANSFORM[i]:
            data[:, :, i, :, :] = torch.log(data[:, :, i, :, :] + 1)

    # Normalize data.
    mean = torch.tensor(_MEAN)[None, None, :, None, None]
    std = torch.tensor(_STD)[None, None, :, None, None]
    data = (data - mean) / std

    # Choose the pre-defined index in bands.
    data = data[:, :, index, :, :]

    # Permute temporal dimensions.
    data = torch.permute(data, (0, 2, 3, 1))

    # Collapse temporal and band dimensions.
    data = torch.flatten(data, start_dim=0, end_dim=2)

    return data

def label_preprocess(data):
    data = data.float()

    # Clip the labels.
    data = torch.clamp(data, 0, _CLIP)
    
    # Replace NaN values with -1 which are later going to be replaced with the label 0 and ignored from the criterion. 
    data[torch.isnan(data)] = -1

    # Collapse dimensions.
    data = torch.flatten(data)

    return data

def bin_label(data):
    bins = torch.tensor(_BINS)
    bins = bins.to(device)
    data = torch.bucketize(data, bins)

    return data

# Fix random seed.
torch.manual_seed(0)

# Load dataset.
dataset = rionegrodata.RioNegroData(
    root="/scratch/tgokce/algal-bloom/data",
    reservoir="palmar",
    window_size=1,
    prediction_horizon=1,
    input_size=224,
)

# Train split. This will sample random bounding boxes from the dataset of size input_size.
train_roi = BoundingBox(
    minx=dataset.roi.minx,
    maxx=dataset.roi.maxx,
    miny=dataset.roi.miny,
    maxy=dataset.roi.maxy,
    mint=dataset.roi.mint,
    maxt=datetime(2021, 12, 31).timestamp(),
)
train_sampler = RandomGeoSampler(
    dataset.data_bio_unprocessed,
    size=float(dataset.input_size),
    length=1000,  # Number of iterations in one epoch.
    roi=train_roi,
)
train_loader = DataLoader(
    dataset, batch_size=16, num_workers=8, sampler=train_sampler
)

# Test split. This will sample the original images without cropping from the dataset.
test_roi = BoundingBox(
    minx=dataset.roi.minx,
    maxx=dataset.roi.maxx,
    miny=dataset.roi.miny,
    maxy=dataset.roi.maxy,
    mint=datetime(2021, 12, 31).timestamp(),
    maxt=dataset.roi.maxt,
)
test_sampler = GridGeoSampler(
    dataset.data_bio_unprocessed,
    size=(dataset.roi.maxy - dataset.roi.miny, dataset.roi.maxx - dataset.roi.minx),
    stride=1,
    roi=test_roi,
    units=Units.CRS,
)

test_loader = DataLoader(dataset, batch_size=1, num_workers=2, sampler=test_sampler)

# Assessed data modalities.
modalities =["chll", "turbidity", "cdom", "water-temp", "mean-air-temp", "mean-cloud-coverage", "precipitation-sum", "mean-radiation", "mean-relative-humidity", "mean-u-wind", "mean-v-wind"]

for index in range(len(modalities)):

    # For collection of statistics.
    results = []

    # Create a subdirectory for the assessed data modality. 
    os.mkdir(modalities[index])
    
    # Change directory to save results.
    os.chdir(modalities[index])

    # Create model.
    model = LinearClassifier().to(device)

    # Criterion to calculate loss values.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, weight=_WEIGHTS)

    # Optimizer to calculate the gradients.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # TRAIN THE MODEL

    all_loss = []

    # Train the neural network with all the training data for the chosen number of cycles.
    for epoch in range(25):

        temp_loss = []

        for x_image, _, y in train_loader:

            # Preprocess the input and the ground truth data.
            # x_image = input_preprocess(x_image) 

            x_image = modality_preprocess(x_image, index)

            y = label_preprocess(y)
            
            # Move data to device.
            x_image = x_image.to(device)
            y = y.to(device)

            # Forward pass.
            y_pred = model(x_image)
            
            # Bin ground truth values to dedicated intervals for the classification task.
            y = bin_label(y)

            # Start backpropagation.
            loss = criterion(y_pred, y)
            loss.backward()

            temp_loss.append(loss.item())

            # Clean-up memory.
            del x_image, y, y_pred, loss

            # Calculate gradient.
            optimizer.step()

            # Gradients have to be equal to 0 after each epoch.
            optimizer.zero_grad()

        all_loss.append(np.mean(temp_loss))
        results.append(f"Epoch: {epoch}, loss: {np.mean(temp_loss)}")

    # Plot that shows how loss decreases after each epoch (training).
    plt.ylabel('Loss value')
    plt.xlabel('Epoch')
    plt.title("Decrease of loss through the training process with each epoch")
    plt.plot(all_loss)
    plt.savefig("loss-plot.png")
    plt.clf()

    # TEST THE MODEL

    class_correct = list(0. for i in range(len(_BINS) + 1))
    class_total = list(0. for i in range(len(_BINS) + 1))

    test_loss = []

    # Using no_grad to prevent calculating the gradients for memory efficiency.
    with torch.no_grad():

        # Iterate through training set. Will stop after one epoch.
        for x_image, _, y in test_loader:

            # Perform the same steps explained in training process for testing the model.
            # x_image = input_preprocess(x_image)

            x_image = modality_preprocess(x_image, index)
            y = label_preprocess(y)

            x_image = x_image.to(device)
            y = y.to(device)

            y_pred = model(x_image)

            y = bin_label(y)

            _, predicted = torch.max(y_pred.data, 1)

            # Compute test loss.
            test_loss.append(criterion(y_pred, y).item())

            c = (predicted == y).squeeze()

            for i in range(len(y)):
                # Identify the label.
                label = y[i]
                # If the ground truth is the same as the predicted value, increment the count of correct prediction for the label.
                class_correct[label] += c[i].item()
                # Increment occurence of the label nonetheless.
                class_total[label] += 1

            # Clean-up memory.
            del x_image, y, y_pred

    # Log accuracy metrics.
    for i in range(len(_BINS) + 1):
        if class_total[i] != 0:
            if i == 0:
                results.append(f"Accuracy for the ignored values (less than 0 or NaNs) is: {100 * class_correct[i] / class_total[i]},  correctly predicted: {class_correct[i]}, total encounters: {class_total[i]}")
            elif i == len(_BINS):
                results.append(f"Accuracy of the bin interval {_BINS[(i - 1)]}+: {100 * class_correct[i] / class_total[i]}, correctly predicted: {class_correct[i]}, total encounters: {class_total[i]}")
            else:
                results.append(f"Accuracy of the bin interval {_BINS[(i - 1)]} - {_BINS[(i)]}: {100 * class_correct[i] / class_total[i]}, correctly predicted: {class_correct[i]}, total encounters: {class_total[i]}")
        else:
            results.append(f"No accuracy (no instance of the class encountered).")

    results.append(f"Total accuracy: {100 * sum(class_correct[1:]) / sum(class_total[1:])}")

    # Plot test loss for each batch in the test loader.
    plt.ylabel('Loss value')
    plt.xlabel('Iteration in the test data')
    plt.title("Decrease of loss through the testing process")
    plt.plot(test_loss)
    plt.savefig("test-loss-plot.png")
    plt.clf()
    
    # Create a file for logging results of the chosen data modality. 
    with open("statistics.txt", "w+") as f:

        # Write the logged data into the file.
        f.write('\n'.join(results)) 
        
        # Close the file.
        f.close()

    # Change directory to parent to begin with the next data modality.
    os.chdir("..")