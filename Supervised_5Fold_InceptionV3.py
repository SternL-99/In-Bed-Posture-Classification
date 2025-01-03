import csv
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from itertools import product
import numpy as np

# Step 1: Load and preprocess the dataset
def load_data(data_dir, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

# Step 2: Define the Inception V3 model with tunable hyperparameters
def build_model(num_classes, num_layers, units_per_layer, dropout_rate, input_size=224):
    model = models.inception_v3(pretrained=False, aux_logits=False)  # Set aux_logits=False for simpler architecture

    # Freeze the convolutional layers (optional)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier of Inception V3
    classifier_layers = []
    in_features = model.fc.in_features  # Input size of the final fully connected layer in Inception V3

    # First FC layer
    classifier_layers.append(nn.Linear(in_features, units_per_layer))
    classifier_layers.append(nn.ReLU())
    classifier_layers.append(nn.Dropout(dropout_rate))

    # Additional FC layers based on num_layers
    for _ in range(num_layers - 1):  # Skip the first layer already added
        classifier_layers.append(nn.Linear(units_per_layer, units_per_layer))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout_rate))

    # Final output layer
    classifier_layers.append(nn.Linear(units_per_layer, num_classes))

    # Replace the classifier in Inception V3
    model.fc = nn.Sequential(*classifier_layers)

    return model


# Step 3: Train the model with Early Stopping and Learning Rate Scheduler
# Step 3: Train the model and save epoch metrics to a CSV
def train_and_evaluate(model, train_loader, val_loader, epochs, device, patience=5, output_dir=r"/cluster/projects/kite/LindsayS/metrics_directory_InceptionV3", csv_filename=r"/cluster/projects/kite/LindsayS/epoch_metrics_InceptionV3.csv"):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    csv_path = os.path.join(output_dir, csv_filename)

    # Open the CSV file and write the header only once
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Learning Rate"])

    # Define class weights and optimization parameters
    class_weights = torch.tensor([1.75, 0.875, 0.875, 0.875, 1.75, 0.7], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct = 0.0, 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_losses.append(train_loss)
        train_accuracies.append(correct / total)

        # Validation phase
        model.eval()
        running_loss, correct = 0.0, 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_losses.append(val_loss)
        val_accuracies.append(correct / total)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Print and save metrics
       # Retrieve and print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']  # Access the updated learning rate
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracies[-1]:.4f}, LR: {current_lr:.6f}")

        # Append the results to the CSV file
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, train_loss, train_accuracies[-1], val_loss, val_accuracies[-1], current_lr])

    return train_losses, val_losses, train_accuracies, val_accuracies



# Step 4: Save metrics to CSV and plots
def save_metrics_and_plots(metrics, param_dict, fold, output_dir):
    params_str = "_".join([f"{k}_{v}" for k, v in param_dict.items()])
    fold_dir = os.path.join(output_dir, f"{params_str}_fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Save metrics to CSV
    csv_file = os.path.join(fold_dir, "metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])
        for i, (train_loss, val_loss, train_acc, val_acc) in enumerate(
            zip(metrics["train_losses"], metrics["val_losses"], metrics["train_accuracies"], metrics["val_accuracies"])
        ):
            writer.writerow([i + 1, train_loss, val_loss, train_acc, val_acc])

    # Save loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["train_losses"], label="Training Loss")
    plt.plot(metrics["val_losses"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, "loss_plot.png"))
    plt.close()

    # Save accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["train_accuracies"], label="Training Accuracy")
    plt.plot(metrics["val_accuracies"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, "accuracy_plot.png"))
    plt.close()

# Step 5: Perform 5-fold cross-validation with grid search
def cross_validate_and_grid_search(data_dir, param_grid, device, output_dir=r"/cluster/projects/kite/LindsayS/metrics_directory_InceptionV3"): #"/cluster/projects/kite/LindsayS/metrics_directory_InceptionV3"
    dataset = load_data(data_dir)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all combinations of parameters
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Testing parameters: {param_dict}")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
            print(f"Fold {fold}")

            # Split dataset
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=param_dict['batch_size'], shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=param_dict['batch_size'], shuffle=False)

                        # Print the size of the train and validation subsets
            print(f"Train subset size: {len(train_subset)}")
            print(f"Validation subset size: {len(val_subset)}")

            # Inspect the first batch from the DataLoader
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
            print(f"First batch from train loader - Inputs shape: {train_batch[0].shape}, Labels shape: {train_batch[1].shape}")

            val_iter = iter(val_loader)
            val_batch = next(val_iter)
            print(f"First batch from val loader - Inputs shape: {val_batch[0].shape}, Labels shape: {val_batch[1].shape}")

            print(f"dataset: {dataset.classes}")
            num_classes=len(dataset.classes)
            print(f"num class: {num_classes}")

            # Build and train the model
            model = build_model(
                num_classes=len(dataset.classes),
                num_layers=param_dict['num_layers'],
                units_per_layer=param_dict['units_per_layer'],
                dropout_rate=param_dict['dropout_rate']
            ).to(device)

            metrics = {}
            print(f"Training model with parameters: {param_dict}")
            metrics["train_losses"], metrics["val_losses"], metrics["train_accuracies"], metrics["val_accuracies"] = train_and_evaluate(
                model, train_loader, val_loader, param_dict['epochs'], device
            )

            # Save metrics and plots
            save_metrics_and_plots(metrics, param_dict, fold, output_dir)

# Main script
if __name__ == "__main__":
    data_dir = r"/cluster/projects/kite/LindsayS/Classes"  # Replace with your dataset path
    output_dir=r"/cluster/projects/kite/LindsayS/metrics_directory_InceptionV3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Define the parameter grid
    param_grid = {
        'num_layers': [1, 2],
        'units_per_layer': [128, 256],
        'dropout_rate': [0.2, 0.3, 0.5],
        'batch_size': [16, 32, 64],
        'epochs': [10, 15]
    }

    cross_validate_and_grid_search(data_dir, param_grid, device)
