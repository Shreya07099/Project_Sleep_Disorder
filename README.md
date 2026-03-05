# Project_Sleep_Disorder

Author: Shreya Nair, VIT Vellore

## Sleep Disorder Analysis and Classification

### Project Overview
This project analyzes sleep disorder data from multiple subjects (AP01 to AP05), processing raw physiological signals (e.g., flow events, nasal flow, sleep profile, SPO2, thorac) to create datasets, train machine learning models for classification, and generate visualizations. It focuses on detecting sleep disorders through data preprocessing, feature extraction, and predictive modeling.

### Key Features
- **Data Processing**: Converts raw text-based signals into structured CSV datasets and NumPy arrays for model input.
- **Model Training**: Trains classification models on preprocessed data to predict sleep disorder categories.
- **Visualization**: Generates plots and PDFs for data insights and model performance.
- **Modular Scripts**: Separate scripts for dataset creation, CSV formation, training, and visualization.

### Technical Implementation
- **Data Pipeline**: Raw data ingestion, filtering, and transformation into labeled datasets.
- **Machine Learning**: Classification models trained on processed features with evaluation metrics.
- **Output Generation**: Automated creation of visualizations and reports.

### Technical Stack
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
- **Tools**: Python, Jupyter Notebook.

### Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run scripts in the `Scripts/` directory.

### Usage- How to work with this 


### Files and Folders
- **Data/**: Raw data files for each subject (AP01-AP05), including flow events, nasal flow, sleep profiles, SPO2, and thorac signals.
- **Dataset/**: Final CSV datasets for each subject.
- **Filtered_Data_Plots/**: Plots from data filtering.
- **models/**: Trained model files.
- **Processed_Dataset/**: Preprocessed CSVs and NumPy arrays (X and y) for each subject.
- **Scripts/**: Core scripts:
  - [create_dataset.py](Scripts/create_dataset.py): Dataset creation.
  - [csv_formation.py](Scripts/csv_formation.py): CSV formation.
  - [model_training.py](Scripts/model_training.py): Model training.
  - [vis.py](Scripts/vis.py): Visualization.
- **Visualizations_PDFs/**: Generated PDF reports and plots.
- [requirements.txt](requirements.txt): Dependencies.
- [rough.py](rough.py): Miscellaneous code.

