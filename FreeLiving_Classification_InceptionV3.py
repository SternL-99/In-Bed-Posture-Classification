import os
import numpy as np
import cv2
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd

# Define your model architecture
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define layers here (adjust to your architecture)
        self.features = models.inception_v3(pretrained=False)  #algorithm
        self.features.fc = nn.Linear(self.features.fc.in_features, 6)  #6 classes

    def forward(self, x):
        return self.features(x)

# Load the pretrained model
pretrained_data_file = "subject_1_model.pth"
model = YourModel()  # Replace with your model class
state_dict = torch.load(pretrained_data_file)
#print(state_dict.keys())
model.load_state_dict(state_dict, strict=False)
# Convert the model to double (float64) if it was trained using float64
model = model.to(torch.double)
model.eval()

# Assuming input_size is defined in your model or as a separate attribute
input_size = (224, 224)  # Adjust according to your model's input requirements

# Directory settings
input_folder = r"C:\Users\SternL\Desktop\Phase I\RawFiles_Videos_Images_Codes\Labs_RawData\CareLab\Exp3\Sub1"
output_folder = "./output_results/"
os.makedirs(output_folder, exist_ok=True)

for txt_filename in [f for f in os.listdir(input_folder) if f.endswith('.txt')]:
    base_name = os.path.splitext(txt_filename)[0]
    file_reader = np.loadtxt(os.path.join(input_folder, txt_filename), delimiter=",")
    x = pd.read_csv(txt_filename, header=None).values  # Adjust delimiter if necessary
    n = x.shape[0]  # Number of rows

    output_file = os.path.join(output_folder, f"{base_name}_Classification")
    video_filename = os.path.join(output_folder, f"{base_name}_Classification")

    # Video writer setup
    frame_width = 500
    frame_height = 400
    output_video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 1, (frame_width, frame_height))

    # Define the transform to apply to the frames before passing them to the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust based on the model's expected input
    ])

    # Process each frame
    for i in range(4, n):  # Start at 5 
        file_reader = x[i,:][~np.isnan(x[i,:])]
        print(i)
        shape = file_reader.reshape((48, 22))
        means = np.mean(shape, axis=0)

        # Find columns with mean less than 10
        zero_mean = np.where(means < 10)[0]
        zero_mean = np.sort(zero_mean)[::-1]

        # Check adjacent columns and handle zero-mean columns
        for col in zero_mean:
            num_row, num_col = shape.shape
            if col < num_col:
                if ((col > 0 and means[col - 1] >= 10) and (col < num_col - 1 and means[col + 1] >= 10)):
                    shape[:, col] = 0
                elif ((col > 0 and means[col - 1] >= 10) and (col < num_col - 2 and means[col + 2] >= 10)):
                    shape[:, col] = 0
                elif ((col > 1 and means[col - 2] >= 10) and (col < num_col - 1 and means[col + 1] >= 10)):
                    shape[:, col] = 0
            shape = np.hstack([shape, np.zeros((num_row, 1))])

        # Plot and resize frame for model input
        plt.imshow(shape, cmap='viridis', aspect='auto')
        plt.axis('off')
        plt.savefig("temp_frame.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        frame = cv2.imread("temp_frame.png")
        resized_frame = resize(frame, input_size)

        # Apply transformation and make prediction
        frame_tensor = transform(resized_frame).unsqueeze(0)  # Add batch dimension
        # Convert to DoubleTensor
        frame_tensor = frame_tensor.to(torch.double)
        with torch.no_grad():
            output = model(frame_tensor)
        
        # Assuming the output is a classification output (e.g., softmax probabilities)
        classfn = torch.argmax(output, dim=1).item()
        #text = str(classfn)

        # Map the class index to the corresponding label
        if classfn == 0:
            text = "Empty"
        elif classfn == 1:
            text = "Left"
        elif classfn == 2:
            text = "Prone"
        elif classfn == 3:
            text = "Right"
        elif classfn == 4:
            text = "Sit"
        elif classfn == 5:
            text = "Supine"
        else:
            text = "Unknown"  # In case of any unexpected class index

        # Add text to the frame
        annotated_frame = cv2.putText(frame, text, (130, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Video Player", annotated_frame)
        cv2.waitKey(100)  # Pause for 0.1 seconds

        # Write frame to video
        output_video.write(annotated_frame)

        # Write classification to file
        with open(output_file, 'a') as file:
            file.write(f"{text}\n")

    # Cleanup
    cv2.destroyAllWindows()
    output_video.release()
    os.remove("temp_frame.png")
