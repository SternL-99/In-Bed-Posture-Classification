# In-Bed-Posture-Classification
The dataset collected for this study consists of 10 participants lying on a hospital mattress, home mattress, and a home mattress with a foam topper. Unfortunately, the dataset can not be published, however, the raw data consists of csv files with pressure data in a 22x48 matrix, with data collected at a frequnecy of 1 Hz. The data analysis for this experiment consists of two portions: (1) supervised postures of participants lying in the supine, prone, right lateral, and left lateral postures as well as seated positions and an empty bed and (2) unsupervised postures correlating to these classes, refered to as free living.

The "PressureValuesToImages" code converts the pressure data into a visual of the in bed posture.

The "Supervised_5Fold_X" code is used to find the best hyperparameters for each model to then be used for the LOSO model generation.Perfromance metrics are created and saved here for evaluation. X represents the model -- four models have been examined (1) ShuffleNet v2, (2) ResNet50, (3) VGG16, and (4) Inception v3

The "Supervised_LOSO_X" code validates the model by leaving one subject out. The trained model is then saved to be used for the "FreeLiving_Classification_X" code. Perfromance metrics are also created and saved here for evaluation. X represents the model -- four models have been examined (1) ShuffleNet v2, (2) ResNet50, (3) VGG16, and (4) Inception v3

The "FreeLiving_Classification_X" code tests the trained 2D CNN models on the free living data to test the performance of the model on unsupervised postures. X represents the model -- four models have been examined (1) ShuffleNet v2, (2) ResNet50, (3) VGG16, and (4) Inception v3

The "FreeLiving_GANS" code is used to help generalize the dataset through creating synthetic data informed by the supervised and unsupervised data.
