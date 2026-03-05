#this is for signal preprocessing and filtering
#note the pls run the csv_formation.py file first to generate the combined dataframe before using it in this 
#script for preprocessing and filtering.
#this script will filter the input subject data and save a flattened csv file for each subject in the processedd dataset folder
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os
import matplotlib.pyplot as plt
#First we will design a bandpass Butterworth filter to isolate the respiratory frequency range (0.17-0.4 Hz) for both Flow and Thorax signals.
def filter1(subject_id = "AP01"):
    df = pd.read_csv(f"Project_Sleep_Disorder\Dataset\\final_dataset_{subject_id}.csv")
    fs = 32.0       # Sampling freq
    lowcut = 0.17   
    highcut = 0.4   
    order = 4       #4th order
    nyq = fs/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    df['Flow_Filtered'] = filtfilt(b, a, df['Flow'])
    df['Thorax_Filtered'] = filtfilt(b, a, df['Thorax'])
    return df
#Plotting those signals(optional,for visual check)
def plot(sub):
    df = filter1(sub)
    fs=32.0
    start_idx = 80000#can be any index in the dataset
    duration_sec = 120# plotting for 2 min example,can be changed
    end_idx = int(start_idx + (duration_sec * fs)) # Cast to int
    subset = df.iloc[start_idx:end_idx]

    time = np.linspace(0, duration_sec, len(subset))

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, subset['Flow'], color='red', alpha=0.6, label='Raw Flow')
    plt.plot(time, subset['Flow_Filtered'], color='blue', linewidth=1.5, label='Filtered Flow (0.17-0.4Hz)')
    plt.title(f'Nasal Flow: Raw vs. Filtered (1)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(time, subset['Thorax'], color='red', alpha=0.6, label='Raw Thorax')
    plt.plot(time, subset['Thorax_Filtered'], color='green', linewidth=1.5, label='Filtered Thorax (0.17-0.4Hz)')
    plt.title(f'Thorax Effort: Raw vs. Filtered (1))')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()
    
    output_dir = "Project_Sleep_Disorder/Filtered_Data_Plots"
    file_name = f"Signal_Check_{sub}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, dpi=300) #dpi=300 ensures high resolution
    
    
#To plot, just put subject id in plot function call AFTER you have run the csv_formation.py

####PREPROCESSING and SAVING DATA as csv and np array####
# I'm using both csv and npy formats because csv is human readable and can be easily used for quick checks and visualizations, 
# while npy is more efficient for loading into the CNN model
#To check please run the preprocess_and_save function with subject id after running the csv_formation.py 
# and check the csv results in Processed_Dataset Folder
#Since the matrix dimension is (960*3*1820) for each subject, np array is better
#csv for each subject will be flattened out to about 1749120 rows each fro 3d to 2d 

def preprocess_and_save(subject_id = "AP01"):
    df = filter1(subject_id)
    output_dir = "Project_Sleep_Disorder/Processed_Dataset"
    os.makedirs(output_dir, exist_ok=True)
    window_sec=30
    overlap_pct=0.5
    fs=32
    samples_per_window = int(window_sec * fs) # 960
    step_size = int(samples_per_window * (1 - overlap_pct)) # 480 as there is a 50 percent overlap(15 sec in  30 sec window)
    X = []
    y = []
    #Iterate through the dataframe in steps
    for start in range(0, len(df) - samples_per_window, step_size):
            end = start + samples_per_window
            window_df = df.iloc[start:end]
            #Get the 'Target' column values for this window
            #(Target 1 = Hypopnea, Target 2 = Obstructive Apnea, 0 = Normal)
            counts = window_df['Target'].value_counts()
            #Find the most frequent label in this 30s window
            most_frequent_label = counts.idxmax() if not counts.empty else 0#checking if it by mistake came out to be 0 so make it default
            #Apply the 50% overlap rule: 
            #Is the most frequent disorder present for > 15 seconds-check
            if most_frequent_label != 0 and (counts[most_frequent_label] > (samples_per_window / 2)):
                window_label = most_frequent_label
            else:
                window_label = 0 #Label as Normal
                
            #Extract the filtered signals for X
            #Features: Flow, Thorax, SpO2_Interpolated
            signal_data = window_df[['Flow_Filtered', 'Thorax_Filtered', 'SpO2_Interpolated']].values
            
            X.append(signal_data)
            y.append(window_label)
    #now X is of dimension(1820, 960, 3) and y is of dimension(1820,) for each subject
            #print(np.array(X).shape)
            #print(np.array(y).shape)
            #Save the preprocessed data for model training
    npx=np.array(X)
    npy=np.array(y)
    np.save(os.path.join(output_dir, f"X_{subject_id}.npy"), npx)
    np.save(os.path.join(output_dir, f"y_{subject_id}.npy"), npy)
            #save the csv for quick checks and visualizations
    preprocessed_csv_path = os.path.join(output_dir, f"preprocessed_{subject_id}.csv")
    (num_windows, samples_per_window, num_features) = npx.shape
            #flatten 3d to 2d
    flattened_X = npx.reshape(-1, num_features)
            #Create metadata columns
            #We repeat the Window_ID (0-1821) and the Label (0-2) 960 times each so every single row in the CSV is identified
    window_ids = np.repeat(np.arange(num_windows), samples_per_window)
    labels = np.repeat(npy, samples_per_window)
            #Create the DataFrame
    df_export = pd.DataFrame(flattened_X, columns=['Flow', 'Thorax', 'SpO2'])
    df_export['Window_ID'] = window_ids
    df_export['Target'] = labels
    df_export.to_csv(preprocessed_csv_path, index=False)
    

        
for i in range(1,6):
     preprocess_and_save(f"AP0{i}")
     print(f"Preprocessing and saving completed for AP0{i}")