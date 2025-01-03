import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the base directory containing the folders
base_dir = r'C:\Users\SternL\Desktop\Phase I\RawFiles_Videos_Images_Codes\Labs_RawData\HomeLab_Foam\Exp1'
output_dir = "output_images_home"  # Directory to save images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Walk through all folders in the base directory
for subdir, dirs, files in os.walk(base_dir):
    # Only process files if the subdir contains files (skip empty folders)
    if files:
        # Filter out only files that are numbers (or match a specific naming pattern)
        sorted_files = sorted(files, key=lambda f: int(''.join(filter(str.isdigit, f))))

        # Process the first 21 files 
        sorted_files = sorted_files[:21]

        # Iterate through each file in the sorted list of files
        for k, base_file_name in enumerate(sorted_files, start=1):
            full_file_name = os.path.join(subdir, base_file_name)
            
            # Ensure we are working with a valid file
            if os.path.isfile(full_file_name):
                print(f"Processing file: {full_file_name}")
                
                # Read the matrix data
                x = pd.read_csv(full_file_name, header=None).values 
                n = x.shape[0]  # Number of rows

                max_row = min(n, 65) #maximum rows for files
                
                # Process each row from the 5th to the end
                for i in range(4, max_row):  # Python is 0-based, MATLAB is 1-based
                    x_clean = x[i,:][~np.isnan(x[i,:])]
                    
                    # Ensure the cleaned data can be reshaped to 48x22
                    if x_clean.size == 48 * 22:
                        y = x_clean.reshape(48, 22)  # Reshape and transpose to match MATLAB

                    
                        means = np.mean(y, axis=0)  # Calculate means of every column
                        zero_mean = np.where(means < 10)[0]  # Find columns with mean less than 10
                        zero_mean = np.sort(zero_mean)[::-1]  # Sort in descending order

                        num_rows, num_cols = y.shape

                        # Check if adjacent columns have a mean >= 10
                        for col in zero_mean:
                            if (
                                col > 0 and means[col - 1] >= 10 and 
                                col < num_cols - 1 and means[col + 1] >= 10
                            ) or (
                                col > 0 and means[col - 1] >= 10 and 
                                col < num_cols - 2 and means[col + 2] >= 10
                            ) or (
                                col > 0 and means[col - 1] >= 10 and 
                                col < num_cols - 3 and means[col + 3] >= 10
                            ) or (
                                col > 1 and means[col - 2] >= 10 and 
                                col < num_cols - 1 and means[col + 1] >= 10
                            ) or (
                                col > 1 and means[col - 2] >= 10 and 
                                col < num_cols - 2 and means[col + 2] >= 10
                            ) or (
                                col > 1 and means[col - 2] >= 10 and 
                                col < num_cols - 3 and means[col + 3] >= 10
                            ) or (
                                col > 2 and means[col - 3] >= 10 and 
                                col < num_cols - 1 and means[col + 1] >= 10
                            ) or (
                                col > 2 and means[col - 3] >= 10 and 
                                col < num_cols - 2 and means[col + 2] >= 10
                            ) or (
                                col > 2 and means[col - 3] >= 10 and 
                                col < num_cols - 3 and means[col + 3] >= 10
                            ):
                                if col < num_cols:  # Only delete if it is within the valid range
                                    y = np.delete(y, col, axis=1)  # Delete column
                                    y = np.hstack((y, np.zeros((num_rows, 1))))  # Add zero column at the end
                                                
                        # Generate the frame using matplotlib
                        plt.figure(figsize=(2.2, 4.8))  # Adjust figure size for better resolution
                        plt.imshow(y, cmap='viridis', aspect='auto')
                        plt.axis('off')  # Turn off axes
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
                        
                        # Save the image with a unique name
                        image_name = f"{os.path.basename(subdir)}_Exp1_Foam_{k}_frame{i+1}.png"
                        image_path = os.path.join(output_dir, image_name)
                        plt.savefig(image_path, dpi=100)
                        plt.close()
                    else:
                        print(f"Skipping frame {i+1} in {base_file_name}: Invalid reshape size")
