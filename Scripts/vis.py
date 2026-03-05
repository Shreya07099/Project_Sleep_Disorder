#This is th py file for visualizations
#Note that the csv_formation.py file is used to create a combined dataframe with all the relevant data, 
# which can then be used for visualizations in this vis.py file. 
#The csv_formation.py file should be run first to generate the combined dataframe before using it in this vis.py file for plotting and analysis.
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def generate_full_report(data_file):
    participant_name = os.path.basename(data_file).replace('final_dataset_', '').replace('.csv', '')
    output_dir = "Project_Sleep_Disorder\Visualizations_PDFs"
    os.makedirs(output_dir, exist_ok=True)
    #load data the usual
    df = pd.read_csv(data_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    #define window
    window_duration = pd.Timedelta(minutes=5)
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()

    pdf_path = os.path.join(output_dir, f"{participant_name}_Visualisations.pdf")

    with PdfPages(pdf_path) as pdf:
        current_start = start_time
        page_num = 1
        colors = {1:'yellow', 2:'orange'} 
        labels = {1:'Hypopnea', 2:'Obstructive Apnea'}

        while current_start < end_time:
            current_end = current_start + window_duration

            #Slice data for the current 5-minute duration
            mask = (df['Timestamp'] >= current_start) & (df['Timestamp'] < current_end)
            df_page = df.loc[mask]

            if not df_page.empty:
                #create fig
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

                # plot 3 plots
                ax1.plot(df_page['Timestamp'], df_page['Flow'], color='blue', lw=0.6)
                ax1.set_ylabel('Flow (32Hz)')
                ax1.set_title(f"Subject: {participant_name} | Page {page_num} | {current_start.strftime('%H:%M:%S')}| 5 min duration")
                ax1.set_ylim(-350,350)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(current_start, current_end)
                
                #chest movement
                ax2.plot(df_page['Timestamp'], df_page['Thorax'], color='green', lw=0.6)
                ax2.set_ylabel('Thorax (32Hz)')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(-350,350)
                ax2.set_xlim(current_start, current_end)

                # SpO2 (4hz-interpolated to 32hz)
                ax3.plot(df_page['Timestamp'], df_page['SpO2_Interpolated'], color='red', lw=1.2)
                ax3.set_ylabel('SpO2 %')
                ax3.set_ylim(80, 100)
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(current_start, current_end)
                #fill in colours for abnormalities
                for code, color in colors.items():
                     ax1.fill_between(df_page['Timestamp'], -350, 350, 
                     where=(df_page['Target'] == code),
                     color=color, alpha=0.3,label=labels[code])
                ax1.legend(loc='upper right', framealpha=0.8)


                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig) # really imp for memory

            current_start = current_end
            #next page will be the next 5 min
            page_num += 1

    print(f"Report generated: {pdf_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    for i in range(1,6):
        sub_id = f"AP0{i}"
        csv_path = os.path.join(project_root, "Dataset", f"final_dataset_{sub_id}.csv")
        generate_full_report(csv_path)#calling function with path   
    '''sub_id = "AP04"
    csv_path = os.path.join(project_root, "Dataset", f"final_dataset_{sub_id}.csv")
    generate_full_report(csv_path)#calling function with path
'''