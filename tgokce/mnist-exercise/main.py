import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import LinearClassifier

# Create a list of transforms to convert and normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    # Pre-calculated values of the global mean and standard deviation of the MNIST dataset
    transforms.Normalize([0.1307], [0.3081])
])
trainset = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transform)

# Load the train and test data in batches
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testLoader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

# Define 10 classes for 10 different MNIST dataset values, iterate and collect values
classes = np.arange(0, 10, 1)
dataiter = iter(trainLoader)
images, labels = dataiter.next()


def imshow(img):
    # Un-normalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# imshow(torchvision.utils.make_grid(images))

# Create the model

model = LinearClassifier.LinearClassifier()
# Calculate the loss
criterion = torch.nn.CrossEntropyLoss()
# Calculate the gradients
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# TRAIN THE MODEL

all_loss = []

# Training the neural network with all the training data for ten cycles
for epoch in range(10):

    temp_loss = []

    for images, labels in trainLoader:
        # Forward pass
        output = model(images.view(images.shape[0], -1))

        # Start backpropagation
        loss = criterion(output, labels)
        loss.backward()

        temp_loss.append(loss.item())

        # Calculate gradient
        optimizer.step()

        # Gradients have to be equal to 0 after each epoch
        optimizer.zero_grad()

    all_loss.append(np.mean(temp_loss))
    # print(f"Epoch: {epoch}, loss: {np.mean(temp_loss)}")

# Plot that shows how loss decreases after each epoch (training)
# plt.plot(all_loss)
# plt.show()

correct, total = 0, 0

# TEST THE MODEL

# Turn the gradient change off as this is the test
# with torch.no_grad():
#     for images, labels in testLoader:
#         output = model(images.view(images.shape[0], -1))
#
#         # Extract the maximal predicted value
#         _, predicted = torch.max(output.data, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(100 * correct / total)

# Test accuracy for each class

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for images, labels in testLoader:
        output = model(images.view(images.shape[0], -1))
        _, predicted = torch.max(output.data, 1)

        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]}")
