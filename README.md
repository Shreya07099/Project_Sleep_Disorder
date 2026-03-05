# Project_Sleep_Disorder

Author: Shreya Nair, VIT Vellore

## Sleep Disorder Analysis and Classification

### Project Overview
This project analyzes sleep disorder data from multiple subjects (AP01 to AP05), processing raw physiological signals (e.g., flow events, nasal flow, sleep profile, SPO2, thorac) to create datasets, train machine learning models for classification, and generate visualizations. It focuses on detecting sleep disorders through data preprocessing, feature extraction, and predictive modeling.
### Technical Stack
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
- **Tools**: Python, Jupyter Notebook.

### Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run scripts in the `Scripts/` directory.

### Usage- How to work with this(If you want to add new subject data )
1) **Data Preparation**: Place raw data files in the `Data/` folder, name the folder AP(Subject ID), with all text files as structured in the Data folder, the rest of them(Same structure in terms of coloums too)
2) **CSV Formation**: Run the csv_formation.py in Scripts folder, and get the combined data csv file for your subject, which you can use througout the different py files in this folder(SPO2 upsampled to 32Hz from 4Hz in this)
3) **Visualization**: Run the data csv file generated to vis.py, to generate plots of SPO2, Thorac and Flow levels, for your subject data highlighting data, saved in the Visualizationas_PDF
4) **Data_Preprocessing**: In the create_dataset.py script, the subject datas are filtered through a butterworth filter of order 4, run the   plot() funtion with your subject_id to see the plot differences between the raw and the filtered signal,save in the Filtered_Data_Plots Folder.Furthur more, those signals are processed and stored in a numpy array(for model training and testing) and csv(for human readabilty just to check) through a 30 sec window of 50 % overlap(15sec) each in Processed_Dataset Folder. Each window is of 30sec and since the signals are recorded at 32Hz(SPO2 is upsampled from 4Hz to 32Hz) 32- samples recorded every second. Data is recorded every 31.25ms which means there are over 960(30 into 32) samples in a 30 second window. There are three features (SPO2,Thorac and Nasal Flow) making the matrix(960*3)and this is done for 8 hours(30 second window with 50 percent overlap) giving us about 1.5k+(say N) reading depending on the subject readings so the final np array dimension is: (N 960 3) this is flattened out to a 2d csv for human readibility in the Processed_Dataset folder.
5) **Model_Training_and Testing**: The 1D CNN Model is run through this file, with 3 layers, and tested as well; the model was trained on Google Collab for faster runtime and the code is in model_training.py
6) **Results**: Can be seen in the Results folder with the Confusion Matrix.

### Files and Folders
- **Data/**: Raw data files for each subject (AP01-AP05), including flow events, nasal flow, sleep profiles, SPO2, and thorac signals.
- **Dataset/**: Final CSV datasets for each subject.
- **Filtered_Data_Plots/**: Plots from data filtering through butterworth filter
- **models/**: Trained model files.
- **Processed_Dataset/**: Preprocessed CSVs and NumPy arrays (X and y) for each subject.
- **Scripts/**: Core scripts:
  - [create_dataset.py](Scripts/create_dataset.py): Dataset creation.
  - [csv_formation.py](Scripts/csv_formation.py): CSV formation.
  - [model_training.py](Scripts/model_training.py): Model training and testing code
  - [vis.py](Scripts/vis.py): Visualization.
- **Visualizations_PDFs/**: Generated PDF reports and plots.
- [requirements.txt](requirements.txt): Dependencies.
- [rough.py](rough.py): Miscellaneous code.

Note: The severe Class imbalance between Normal(being a lot more thanthe other 2), Hyponea, and obstructive breathing was tried to be handled by first SMOTE, then Undersampling, which gave slightly better result than SMOTE
While Initially, with no applicatoin of such techniques, accuracy was better(around 82%) but however there was a trade off while applying these techniques significantly reduced the overall accuracy as normal class were detected lesser and lesser and increase in detection of disorders did not make much of a difference
Class counts:

Class 0: Normal:7862:87.3%
Class 1: Hypopnea:727: 8.1%
Class 2 :Obstructive:211:2.3%
Total:8,800,100%

But the model i have submitted does not give a higher accuracy(about 50%) about it reports Class 1 and Class 2 better than the previous iterations(can be noted in the confusion matrix submitted in the results folder)
Still needs work




AI Disclosure: Generative AI has been used to technically assist  in the  codes of the preprocessing process of applying SMOTE and under sampling,professional formatting of the README file, and to maintain the structural and memory  efficiency of the Project, but rest assured the author understands each and every concept applied and used and the results

