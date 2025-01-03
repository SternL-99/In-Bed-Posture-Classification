import os
import csv
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, input_size, transform=None):
        input_size = 224
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((input_size, input_size)),  # Resize the images
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
        ])  # Default transform is resize, ToTensor, Normalize
        self.samples = []
        self.class_names = ["Position1", "Position2", "Position3", "Position4", "Position5", "Position6"] #Class Names

        for position_folder in os.listdir(root_dir):
            position_path = os.path.join(root_dir, position_folder)
            if not os.path.isdir(position_path):
                continue

            for subject_folder in os.listdir(position_path):
                subject_path = os.path.join(position_path, subject_folder)
                if not os.path.isdir(subject_path):
                    continue

                position_id = int(position_folder.replace("Position", "")) - 1
                if position_id < 0 or position_id >= 6:
                    raise ValueError(f"Invalid position_id {position_id} found. Expected range: 0 to 5.")
                
                subject_id = int(subject_folder.replace("Sub", ""))
                label = (position_id, subject_id)

                for img_file in os.listdir(subject_path):
                    img_path = os.path.join(subject_path, img_file)
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):  # Add supported formats
                        self.samples.append((img_path, label))

    def get_class_name(self, class_id):
        return self.class_names[class_id]
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply the transformation (Resize, ToTensor, Normalize)
        
        # Convert the label tuple to a tensor
        label = torch.tensor(label, dtype=torch.long)  # Ensure it's a tensor

        position_id = label[0]  # For training, use only the position ID (label[0])
        subject_id = label[1]  # Subject ID (for other purposes, like saving or evaluation)
        
        # Debugging: Print position_id and subject_id to check values
        #print(f"Position ID: {position_id}, Subject ID: {subject_id}")
        
        return image, position_id, subject_id  # Return image, position_id (for training), and subject_id (for other purposes)

# Step 1: Load and preprocess the dataset
def load_data(data_dir, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
    ])
    return CustomDataset

# Step 2: Define the ResNet50 model with tunable hyperparameters
def build_model(num_classes, num_layers, units_per_layer, dropout_rate, input_size=224):
    model = models.resnet50(pretrained=False) 

    # Step 2: Load the weights from a file
    weights_path = r'/cluster/projects/kite/LindsayS/resnet50_weights.pth'  # Path to the weights file
    weights = torch.load(weights_path)

    # Step 3: Load the weights into the model
    model.load_state_dict(weights)

    # Freeze the convolutional layers (optional)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the fully connected (fc) layer of ShuffleNet
    in_features = model.fc.in_features  

    fc_layers = []
    
    # First FC layer
    fc_layers.append(nn.Linear(in_features, units_per_layer))
    fc_layers.append(nn.ReLU())
    fc_layers.append(nn.Dropout(dropout_rate))

    # Additional FC layers based on num_layers
    for _ in range(num_layers - 1):  # Skip the first layer already added
        fc_layers.append(nn.Linear(units_per_layer, units_per_layer))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout_rate))

    # Final output layer
    fc_layers.append(nn.Linear(units_per_layer, num_classes))

    # Replace the original FC layer with the new fully connected layers
    model.fc = nn.Sequential(*fc_layers)

    return model

# Step 3: Train the model with Early Stopping and Learning Rate Scheduler; save epoch metrics to a CSV
def train_and_evaluate(model, train_loader, val_loader, epochs, device, patience=5, output_dir=r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/metrics_directory", csv_filename=r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/epoch_metrics.csv"):
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
        for inputs, position_ids, subject_ids in train_loader:
            inputs, position_ids = inputs.to(device), position_ids.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs, position_ids)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * position_ids.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == position_ids).sum().item()
            total += position_ids.size(0)

        train_loss = running_loss / total
        train_losses.append(train_loss)
        train_accuracies.append(correct / total)

        # Validation phase
        model.eval()
        running_loss, correct = 0.0, 0
        total = 0
        with torch.no_grad():
            for inputs, position_ids, subject_ids in val_loader:
                for position_id in position_ids:
                    if position_id.item() < 0 or position_id.item() >= 6:
                        print(f"Invalid position_id {position_id.item()} found!")
                    else:
                        print(f"Valid position_id {position_id.item()}")
                inputs, position_ids = inputs.to(device), position_ids.to(device)
                outputs = model(inputs).to(device)
                loss = criterion(outputs, position_ids)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == position_ids).sum().item()
                total += position_ids.size(0)

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

# Evaluate model for confusion matrix
def evaluate_model(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, position_ids, subject_ids in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).to(device)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(position_ids.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print("Sample y_true:", y_true[:10])
    print("Sample y_pred:", y_pred[:10])

    return y_pred, y_true

# Save confusion matrix to file
def save_confusion_matrix(y_true, y_pred, subject_id, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    # Save the confusion matrix as an array to a text file
    cm_array_path = os.path.join(output_dir, f"confusion_matrix_subject_{subject_id}.txt")
    np.savetxt(cm_array_path, cm, fmt='%d', delimiter=',')
    print(f"Confusion matrix array saved to {cm_array_path}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)  # Add color for better visualization
    plt.title(f"Confusion Matrix for Subject {subject_id}")
    plt.show()
    cm_path = os.path.join(output_dir, f"confusion_matrix_subject_{subject_id}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix for Subject {subject_id} saved to {cm_path}")

# Save model to file
def save_model(model, subject_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"subject_{subject_id}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model for Subject {subject_id} saved to {model_path}")

# Save training plots to file
def save_training_plot(train_values, val_values, ylabel, subject_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} for Subject {subject_id}")
    plt.legend()
    plot_path = os.path.join(output_dir, f"{ylabel.lower()}_subject_{subject_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"{ylabel} plot for Subject {subject_id} saved to {plot_path}")

# Save metrics summary to CSV
def save_metrics_summary(metrics, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subject ID", "Best Validation Accuracy"])
        writer.writerows(metrics)
    print(f"Metrics summary saved to {output_file}")

# Save logs to file
def save_log(log_text, subject_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"subject_{subject_id}_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(log_text)
    print(f"Log for Subject {subject_id} saved to {log_path}")

# Updated LOSO Cross-Validation with Enhanced Saving
def leave_one_subject_out_with_saving(data_dir, subjects, num_classes, batch_size, epochs, device, output_dirs, input_size):
    dataset = CustomDataset(data_dir, input_size, transform=None)
    metrics_summary = []

    # Limit the subjects b/c of space -- include the following
    #subjects = subjects[0:3]  # Python uses 0-based indexing: 0:3 -> 1:3 -> indexing does not include the last value

    for subject_id in subjects:
        print(f"Subject {subject_id}: Leave-Out")

        # Split dataset by subject (ignoring positions)
        train_idx = [i for i, (_, _, label_subject_id) in enumerate(dataset) if label_subject_id != subject_id]
        val_idx = [i for i, (_, _, label_subject_id) in enumerate(dataset) if label_subject_id == subject_id]

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        print(f"Train subset size: {len(train_subset)}")
        print(f"Validation subset size: {len(val_subset)}")

        # Build and train model as before
        model = build_model(
            num_classes,
            model_params['num_layers'],
            model_params['units_per_layer'],
            model_params['dropout_rate']
        ).to(device)

        train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
            model, train_loader, val_loader, epochs, device
        )

        # Save results (model, metrics, plots, confusion matrices, etc.)
        save_model(model, subject_id, output_dirs['models'])
        save_training_plot(train_losses, val_losses, "Loss", subject_id, output_dirs['training_plots'])
        save_training_plot(train_accuracies, val_accuracies, "Accuracy", subject_id, output_dirs['training_plots'])
        log_text = f"Subject {subject_id} Training Log\nBest Validation Accuracy: {max(val_accuracies):.4f}\n"
        save_log(log_text, subject_id, output_dirs['logs'])
        y_pred, y_true = evaluate_model(model, val_loader, device)
        class_names = [dataset.get_class_name(class_id.item()) for class_id in y_pred]
        #class_names = [f"Position{pos}_Sub{sub}" for pos in range(1, 7) for sub in range(1, 11)]  # Adjust based on actual number of positions and subjects
        save_confusion_matrix(y_true, y_pred, subject_id, class_names, output_dirs['confusion_matrices'])

        metrics_summary.append([subject_id, max(val_accuracies)])

    save_metrics_summary(metrics_summary, os.path.join(output_dirs['results'], "metrics_summary.csv"))


# Main script
if __name__ == "__main__":
    data_dir = r"C/cluster/projects/kite/LindsayS/Classes_Sub"  # Replace with your dataset path
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Replace with your subject IDs
    num_classes = 6
    batch_size = 32
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    output_dirs = {
    "models": r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/models",
    "training_plots": r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/training_plots",
    "confusion_matrices": r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/confusion_matrices",
    "logs": r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/logs",
    "results": r"/cluster/projects/kite/LindsayS/ResNet50_LOSO/analysis_outputs"
    }


    # Define model hyperparameters
    model_params = {
        'num_layers': 2,
        'units_per_layer': 128,
        'dropout_rate': 0.5
    }
    
    leave_one_subject_out_with_saving(data_dir, subjects, num_classes, batch_size, epochs, device, output_dirs, input_size=224)
