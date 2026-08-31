# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1:
Load and preprocess the `defect` and `notdefect` image datasets by resizing them to `224 × 224`, converting them into tensors, and normalizing them.

### STEP 2:
Load the pre-trained VGG19 model with ImageNet weights and freeze the pre-trained layers.

### STEP 3:
Replace the final classifier layer with a layer having 2 output classes and define Cross Entropy Loss and Adam optimizer.

### STEP 4:
Train the model for 10 epochs using the training dataset and calculate training and validation loss.

### STEP 5:
Evaluate the model using test accuracy, confusion matrix, and classification report.

### STEP 6:
Predict individual test images using Softmax and display the actual class, predicted class, and confidence.




## PROGRAM

### Name: Yashwanth asv

### Register Number: 212224230309

```
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import DataLoader,TensorDataset

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

from torchvision import datasets, transforms
dataset_path="C:\\Users\\harie\\OneDrive\\Desktop\\deep learning\\chip_data\\dataset"
train_data=datasets.ImageFolder(root=f"{dataset_path}/train",transform=transform)
test_data=datasets.ImageFolder(root=f"{dataset_path}/test",transform=transform)

def show_sample_images(dataset, num_images=5) :
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images) :
        image, label = dataset[i]
        image = image.permute(1, 2, 0) 
        axes [i].imshow(image)
        axes [i].set_title(dataset.classes[label])
        axes[i].axis ("off")
    plt.show()

show_sample_images(train_data)


print (f"Total number of training samples: {len(train_data)} ") 
print (f"Total number of testing samples: {len (test_data)}")

first_image, label = train_data[0]
print(f"Shape of the first image: {first_image.shape} ")


train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(test_data,batch_size=32,shuffle=False)

num_classes=len(train_data.classes)
print(f"Number of classes: {num_classes}")

import torchvision.models as models

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(model. classifier[6].in_features, num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier[6].parameters(),lr=0.001)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))


def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    
    # Plot training and validation loss
   
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))


train_model(model, train_loader,test_loader)
test_model(model, test_loader)


 def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        

        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(output)
        output2=torch.max(prob,1)
        predicted = torch.argmax(prob, dim=1).item()

    class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

predict_image(model,image_index=55, dataset=test_data)


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot


<img width="718" height="589" alt="image" src="https://github.com/user-attachments/assets/ba678f3c-a078-4da9-b404-12f3c1e2a000" />


## Confusion Matrix

<img width="641" height="462" alt="image" src="https://github.com/user-attachments/assets/bfa13a26-a92d-4156-bf04-5caa614704a2" />


## Classification Report

<img width="541" height="175" alt="image" src="https://github.com/user-attachments/assets/95e5086b-6608-47e0-bbea-c9e779952c03" />


### New Sample Data Prediction

<img width="408" height="357" alt="image" src="https://github.com/user-attachments/assets/6751d114-e1c2-4296-8c0b-12ca5f7f8521" />


<img width="491" height="361" alt="image" src="https://github.com/user-attachments/assets/2acd884f-51d5-45bc-be5f-134b2207edd2" />

## RESULT
The VGG19 transfer learning model was successfully developed to classify chip images into defect and notdefect classes. The model was evaluated using accuracy, confusion matrix, and classification report, and successfully predicted sample images.
