#This is the script to  formed organised csv files of each subject syncronised by time, availble in the dataset folder
#for different subject data, upload the subject files in the Data folder in the required format(given in the README file)
# and change the subject_id variable in this script to generate the combined csv file for that subject.
import os
import pandas as pd
import glob

# Define the base path to the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
subject_id = "AP05"
data_base_path = os.path.join(root_dir, "Data", subject_id)
#Now the prob is for different subjects files are named differently with dates, i could rename them, but not efficient
#ill use glob
flow_file = glob.glob(os.path.join(data_base_path, "*Flow Nasal*.txt"))[0]
spo2_file = glob.glob(os.path.join(data_base_path, "*SPO2*.txt"))[0]
thor_file = glob.glob(os.path.join(data_base_path, "*Thorac*.txt"))[0]
sleep_file = glob.glob(os.path.join(data_base_path, "*Sleep profile*.txt"))[0]
event_file = glob.glob(os.path.join(data_base_path, "*Flow Events*.txt"))[0]



# use the path of each subject file respectively-load nasal airflow(32Hz)
df_flow = pd.read_csv(flow_file, sep=';', skiprows=7, names=['Timestamp', 'Flow'])
# Use .str.replace(',', '.') 
df_flow['Timestamp'] = pd.to_datetime(df_flow['Timestamp'].str.strip().str.replace(',', '.'), format='%d.%m.%Y %H:%M:%S.%f')

# Load SpO2 (4Hz)
df_spo2 = pd.read_csv(spo2_file, sep=';', skiprows=7, names=['Timestamp', 'SpO2'])
df_spo2['Timestamp'] = pd.to_datetime(df_spo2['Timestamp'].str.strip().str.replace(',', '.'), format='%d.%m.%Y %H:%M:%S.%f')

# Replace 0 with NaN so interpolate knows to skip these points
df_spo2['SpO2'] = df_spo2['SpO2'].replace(0, pd.NA)
df_spo2['SpO2'] = pd.to_numeric(df_spo2['SpO2'], errors='coerce')

df_flow = df_flow.sort_values('Timestamp')
df_spo2 = df_spo2.sort_values('Timestamp')

# merge
df_combined = pd.merge(df_flow, df_spo2, on='Timestamp', how='left')


#ignore the periods where the sensor was at 0
df_combined['SpO2_Interpolated'] = df_combined['SpO2'].interpolate(method='linear', limit_direction='both')

# Check a 1-second slice to see the smooth transition
#print(df_combined[['Timestamp', 'SpO2', 'SpO2_Interpolated']].head(50))
#########Merge Chest movement##############
df_thorac = pd.read_csv(thor_file, sep=';', skiprows=7, names=['Timestamp', 'Thorax'])
df_thorac['Timestamp'] = pd.to_datetime(df_thorac['Timestamp'].str.strip().str.replace(',', '.'), format='%d.%m.%Y %H:%M:%S.%f')

# Merge with Dataframe (which already has Flow and SpO2)
#use left to ensure to stay on the 32Hz grid
df = pd.merge(df_combined, df_thorac, on='Timestamp', how='left')
#print(df.head(10))
###########Adding Sleep Profile##############
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Sleep Profile loaded every 30 seconds
df_sleep = pd.read_csv(sleep_file, sep=';', skiprows=7, names=['Timestamp', 'Stage'])
df_sleep['Timestamp'] = pd.to_datetime(df_sleep['Timestamp'].str.strip(), format='%d.%m.%Y %H:%M:%S,%f')

# Sort both by Timestamp (already sorted but still)
df_tocombineto= df.sort_values('Timestamp')
df_sleep = df_sleep.sort_values('Timestamp')

#Merge Sleep Stages into the 32Hz timeline
#because sleep stages are labeled every 30 seconds, we want to fill all the 32Hz rows between those labels with the same stage until the next label comes in.
#'backward' direction ensures a stage labeled at 21:00:00  fills all 32Hz rows until the next label at 21:00:30
df_combined = pd.merge_asof(df_tocombineto, df_sleep, on='Timestamp', direction='backward')

#Entering Sleep Disorder Label###
#these are already given- they are the supervised label-for training remove karna

df_combined['Timestamp'] = pd.to_datetime(df['Timestamp'])
#all rows start with 0 (no disorder) and we will update to 1 or 2 based on the events file
df_combined['Target'] = 0 

# Load the Flow Events file
# skiprows 5 here not 7, each file structure should be the same
df_events = pd.read_csv(event_file, sep=';', skiprows=5, 
                        names=['Time_Range', 'Duration', 'Disorder', 'Stage'])

# map using dict
disorder_mapping = {
    'Hypopnea': 1,
    'Obstructive Apnea': 2
}

#Process each event window
for _, row in df_events.iterrows():
    try:
        #Split the range at -(given in the data start-end)
        time_parts = str(row['Time_Range']).split('-')
        start_str = time_parts[0].strip()
        #Extract the date from the startstr to apply to end
        date_prefix = start_str.split(' ')[0] 
        #create end string by combining date prefix with end time part
        end_str = f"{date_prefix} {time_parts[1].strip()}"
        #Convert to datetime
        start_dt = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S,%f')
        end_dt = pd.to_datetime(end_str, format='%d.%m.%Y %H:%M:%S,%f')
        # Get the label code
        disorder_name = str(row['Disorder']).strip()
        label_code = disorder_mapping.get(disorder_name, 0)
        # Apply the label code(0,1,2)to the duration of the disorder(1,2) or normal(0)
        mask = (df_combined['Timestamp'] >= start_dt) & (df_combined['Timestamp'] <= end_dt)
        df_combined.loc[mask, 'Target'] = label_code
        
    except Exception as e:
        #Skip rows that don't match format(like headers)
        continue

output_dir = os.path.join(root_dir, "Dataset")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
saved_name = f"final_dataset_{subject_id}.csv"
output_path = os.path.join(output_dir, saved_name)
df_combined.to_csv(output_path, index=False)


