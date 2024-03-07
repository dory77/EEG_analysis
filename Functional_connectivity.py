#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:55:03 2024

@author: monir
"""

#%%

def calculate_psd_results(folder_path):
    psd_results = []

    for eeg_file in glob.glob(folder_path + '/*'):
        #%%%%%% étape 1: création d'époques de durées identiques
        eeg = mne.io.read_raw_fif(eeg_file)
        eeg.load_data()
        #Divide continuous raw data into equal-sized consecutive epochs, Creating Fixed-Length Epochs.
        epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)

        #%%%%%% étape 2: calcul des puissances de chaque bande pour toutes les époques et tous les canaux
        #Calculating Power Spectral Density (PSD)
        # frequency_bands: A dictionary defining frequency bands of interest
        frequency_bands = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 30], "gamma": [30, 45]}
        spectrum = epochs.compute_psd("welch", average="mean", picks="eeg", fmin=0.5, fmax=45.0) # 91 epochs × 19 channels × 90 freqs, 0.5-45.0 Hz
        # Normalizing the PSDs
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds = psds / np.sum(psds, axis=-1, keepdims=True) # normalize the PSDs

        bandpower = []  #Calculating Band Powers
        for fmin, fmax in frequency_bands.values():
            psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            bandpower.append(psds_band.reshape(len(psds), -1))   
        bandpower_global_means = [np.mean(band) for band in bandpower]
        delta_power, theta_power, alpha_power, beta_power, gamma_power = bandpower_global_means
        # Extract subject ID from the filename
        subject_id = os.path.basename(eeg_file).split('_')[1].split('.')[0]


        # Append results to the list
        psd_results.append([subject_id, delta_power, theta_power, alpha_power, beta_power, gamma_power])

    psd_results = pd.DataFrame(psd_results, columns=['subject_id', 'delta', 'theta', 'alpha', 'beta', 'gamma'])
    psd_results['subject_id'] = pd.to_numeric(psd_results['subject_id'], errors='coerce')
    psd_results.sort_values(by='subject_id', inplace=True)
    psd_results.set_index('subject_id', inplace=True)
    
    return psd_results



#%%

# Provide the function with desired dataframe
# delta_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,0]) #make a df out of a list for example
import numpy as np
import pandas as pd

def connectivity_calculator(connectivity_dataframe, electrode_names=None, scalp_regions=None):
    """
    Calculate mean connectivity values across specified scalp regions.

    Parameters:
    - connectivity_dataframe (pd.DataFrame): DataFrame containing connectivity values between electrodes.
    - electrode_names (list, optional): List of electrode names. Defaults to None.
    - scalp_regions (dict, optional): Dictionary mapping scalp region names to lists of corresponding electrode names.
                                      Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing mean connectivity values between scalp regions.

    Example:
    # Assuming connectivity_df is a DataFrame containing connectivity values
    mean_connectivity = connectivity_calculator(connectivity_df)
    """
    if electrode_names is None:
        electrode_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3',
                           'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    if scalp_regions is None:
        scalp_regions = {
            'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
            'occipital': ['O1', 'O2'],
            'parietal': ['P3', 'P4', 'Pz'],
            'central': ['C3', 'C4', 'Cz'],
            'temporal': ['T3', 'T4', 'T5', 'T6', 'F7', 'F8']
        }
    
    connectivity_dataframe.columns = electrode_names
    connectivity_dataframe.index = electrode_names
    
    mean_df = pd.DataFrame(index=scalp_regions.keys(), columns=scalp_regions.keys())
    
    for region_row, electrodes_row in scalp_regions.items():
        for region_col, electrodes_col in scalp_regions.items():
            mean_values = connectivity_dataframe.loc[electrodes_row, electrodes_col].mean().mean()
            mean_df.loc[region_row, region_col] = mean_values
    
    # Convert DataFrame to numeric type
    mean_df = mean_df.apply(pd.to_numeric)
    
    # Ensure symmetry
    mean_df[mean_df == 0] = np.nan
    for i in range(len(mean_df)):
        for j in range(len(mean_df.columns)):
            # Check if the value is NaN
            if pd.isnull(mean_df.iloc[i, j]):
                # Check the corresponding cell in the symmetric position
                if not pd.isnull(mean_df.iloc[j, i]):
                    # Fill the NaN cell with the corresponding non-NaN value
                    mean_df.iloc[i, j] = mean_df.iloc[j, i]
    
    return mean_df





#%%


def plot_heatmap(df, frequency_band, index):
    """
    Plot a heatmap of connectivity values for a given frequency band.

    Parameters:
    - df (pd.DataFrame): DataFrame containing connectivity values.
    - frequency_band (str): Frequency band for which the heatmap is plotted.
    - index (int): Index of the plot (for labeling purposes).

    Returns:
    None

    Example:
    # Assuming df contains connectivity values for a specific frequency band
    plot_heatmap(df, 'Theta', 1)
    
    """
    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(df, dtype=bool))
    np.fill_diagonal(mask, False)
    ax = plt.axes()
    ax.set_title(f'Heatmap {index} - {frequency_band}')
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax, vmin=0.05, vmax=0.45) 

    # Define the filename for the plot
    filename = f'heatmap_{index}_{frequency_band}.png'
    
    # Check if the plot file already exists
    if not os.path.exists(filename):
        plt.savefig(filename)  # Save the plot as an image file
    plt.show()



#%%

from pycrostates.metrics import silhouette_score, calinski_harabasz_score, dunn_score, davies_bouldin_score
from mne.channels import make_standard_montage
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import matplotlib
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks, resample, apply_spatial_filter
from pycrostates.datasets import lemon
from pycrostates.cluster import ModKMeans
import pycrostates
from mne.viz import plot_ch_adjacency, plot_topomap
from mne.channels import find_ch_adjacency
from mne.datasets import fetch_fsaverage
from mne.datasets import eegbci
import mne
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import re
from mne_connectivity import spectral_connectivity_epochs 
from mne_connectivity.viz import plot_sensors_connectivity
from mne.datasets import sample
import os
import glob
import pandas as pd
import mne
import seaborn as sns
import numpy as np
from IPython.display import display

%matplotlib qt
matplotlib.use('Qt5Agg')
print("mne version", mne.__version__)  # mne version 1.1.0




def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


montage = make_standard_montage('standard_1020')  # montage.plot()

#ord_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3','C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

montage.plot(kind="3d", show=False)


#%% Control Group

pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv') #phenotype data
pheno_ctrl = pheno[pheno['status']=='ctrl']

input_ctrl_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_controls' #EEG recording
psd_results_ctrl = calculate_psd_results(input_ctrl_group)


# Extract the numeric part of the subject_id column
pheno_ctrl['subject_id'] = pheno_ctrl['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_ctrl['subject_id'] = pd.to_numeric(pheno_ctrl['subject_id'])
merged_ctrl = pheno_ctrl.merge(psd_results_ctrl, left_on='subject_id', right_index=True)




#%%


wpli_results_ctrl = []                              # a list for storing the conncetivity results: # 11 subjects, #19 electrods, #5 frequecy bands
delta_conn_ctrl = []                                # separating based of frequecy bands
theta_conn_ctrl = []
alpha_conn_ctrl = []
beta_conn_ctrl = []
gamma_conn_ctrl = []                                # each list has 11 subjects, connectivity between 19 * 19 electrods 


for subject in pheno_ctrl['subject_id']:
    
    subject_id_str = 'subject_' + str(subject) + '.fif'
    eeg_file = os.path.join(input_ctrl_group, subject_id_str)
    #print(eeg_file)
    # Load EEG data for the subject
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)

    
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)
    
    # Compute connectivity
    conn_ctrl = spectral_connectivity_epochs(epochs, method='wpli', fmin=(0.5,4,8,12,30), fmax=(4,8,12,30,45),
                                        faverage=True).get_data(output='dense')
    
    # Append connectivity results to the list
    wpli_results_ctrl.append(conn_ctrl)   
    
    
    delta_conn_ctrl.append(conn_ctrl[:,:,0])
    theta_conn_ctrl.append(conn_ctrl[:,:,1])
    alpha_conn_ctrl.append(conn_ctrl[:,:,2])
    beta_conn_ctrl.append(conn_ctrl[:,:,3])
    gamma_conn_ctrl.append(conn_ctrl[:,:,4])



delta_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,0]) #make a df out of a list
theta_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,1])
alpha_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,2])
beta_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,3])
gamma_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,4])


# Datafram of connectivity values for each frequency band
delta = connectivity_calculator(delta_conn_ctrl_df)
theta = connectivity_calculator(theta_conn_ctrl_df)
alpha = connectivity_calculator(alpha_conn_ctrl_df)
beta = connectivity_calculator(beta_conn_ctrl_df)
gamma = connectivity_calculator(gamma_conn_ctrl_df)


dataframes = [delta, theta, alpha, beta, gamma]

frequency_bands = ['delta_ctrl_df', 'theta_ctrl_df','alpha_ctrl_df','beta_ctrl_df','gamma_ctrl_df']  # Assuming the order of the dataframes corresponds to these bands

for i, (df, frequency_band) in enumerate(zip(dataframes, frequency_bands), 1):
    plot_heatmapp(df, frequency_band, i)
    
   
#%% Deletion group

pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv')
pheno_del = pheno[pheno['status']=='del']

input_del_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_del'
psd_results_del = calculate_psd_results(input_del_group)

#%% 


wpli_results_del = []

# Extract the numeric part of the subject_id column
pheno_del['subject_id'] = pheno_del['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_del['subject_id'] = pd.to_numeric(pheno_del['subject_id'])
merged_del = pheno_del.merge(psd_results_del, left_on='subject_id', right_index=True)



delta_conn_del = []
theta_conn_del = []
alpha_conn_del = []
beta_conn_del = []
gamma_conn_del = []

for subject in pheno_del['subject_id']:
    
    subject_id_str = 'subject_' + str(subject) + '.fif'
    eeg_file = os.path.join(input_del_group, subject_id_str)
    #print(eeg_file)
    # Load EEG data for the subject
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)

    
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)
    
    # Compute connectivity
    conn_del = spectral_connectivity_epochs(epochs, method='wpli', fmin=(0.5,4,8,12,30), fmax=(4,8,12,30,45),
                                        faverage=True).get_data(output='dense')
    
    # Append connectivity results to the list
    wpli_results_del.append(conn_del)
    
    
    delta_conn_del.append(conn_del[:,:,0])
    theta_conn_del.append(conn_del[:,:,1])
    alpha_conn_del.append(conn_del[:,:,2])
    beta_conn_del.append(conn_del[:,:,3])
    gamma_conn_del.append(conn_del[:,:,4])



delta_conn_del_df = pd.DataFrame(conn_del[:,:,0]) #make a df out of a list
theta_conn_del_df = pd.DataFrame(conn_del[:,:,1])
alpha_conn_del_df = pd.DataFrame(conn_del[:,:,2])
beta_conn_del_df = pd.DataFrame(conn_del[:,:,3])
gamma_conn_del_df = pd.DataFrame(conn_del[:,:,4])



# Datafram of connectivity values for each frequency band
delta_del_df = connectivity_calculator(delta_conn_del_df)
theta_del_df = connectivity_calculator(theta_conn_del_df)
alpha_del_df = connectivity_calculator(alpha_conn_del_df)
beta_del_df = connectivity_calculator(beta_conn_del_df)
gamma_del_df = connectivity_calculator(gamma_conn_del_df)



dataframes = [delta_del_df, theta_del_df, alpha_del_df, beta_del_df,gamma_del_df]

frequency_bands = ['delta_del_df', 'theta_del_df','alpha_del_df','beta_del_df','gamma_del_df']  # Assuming the order of the dataframes corresponds to these bands

for i, (df, frequency_band) in enumerate(zip(dataframes, frequency_bands), 1):
    plot_heatmap(df, frequency_band, i)
    
    
#%% Deletion + Scz group



pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv')
pheno_del_scz = pheno[pheno['status']=='del_scz']

input_del_scz_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_del_scz'
psd_results_del_scz = calculate_psd_results(input_del_scz_group)


wpli_results_del_scz = []

# Extract the numeric part of the subject_id column
pheno_del_scz['subject_id'] = pheno_del_scz['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_del_scz['subject_id'] = pd.to_numeric(pheno_del_scz['subject_id'])
merged_del_scz = pheno_del_scz.merge(psd_results_del_scz, left_on='subject_id', right_index=True)


delta_conn_del_scz = []
theta_conn_del_scz = []
alpha_conn_del_scz = []
beta_conn_del_scz = []
gamma_conn_del_scz = []

for subject in pheno_del_scz['subject_id']:
    
    subject_id_str = 'subject_' + str(subject) + '.fif'
    eeg_file = os.path.join(input_del_scz_group, subject_id_str)
    #print(eeg_file)
    # Load EEG data for the subject
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)

    
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)
    
    # Compute connectivity
    conn_del_scz = spectral_connectivity_epochs(epochs, method='wpli', fmin=(0.5,4,8,12,30), fmax=(4,8,12,30,45),
                                        faverage=True).get_data(output='dense')
    
    # Append connectivity results to the list
    wpli_results_del_scz.append(conn_del_scz)
    
    
    delta_conn_del_scz.append(conn_del_scz[:,:,0])
    theta_conn_del_scz.append(conn_del_scz[:,:,1])
    alpha_conn_del_scz.append(conn_del_scz[:,:,2])
    beta_conn_del_scz.append(conn_del_scz[:,:,3])
    gamma_conn_del_scz.append(conn_del_scz[:,:,4])
    




delta_conn_del_scz_df = pd.DataFrame(conn_del_scz[:,:,0]) #make a df out of a list
theta_conn_del_scz_df= pd.DataFrame(conn_del_scz[:,:,1])
alpha_conn_del_scz_df = pd.DataFrame(conn_del_scz[:,:,2])
beta_conn_del_scz_df = pd.DataFrame(conn_del_scz[:,:,3])
gamma_conn_del_scz_df = pd.DataFrame(conn_del_scz[:,:,4])

# Datafram of connectivity values for each frequency band
delta_del_scz_df  = connectivity_calculator(delta_conn_del_scz_df)
theta_del_scz_df = connectivity_calculator(theta_conn_del_scz_df)
alpha_del_scz_df = connectivity_calculator(alpha_conn_del_scz_df)
beta_del_scz_df = connectivity_calculator(beta_conn_del_scz_df)
gamma_del_scz_df = connectivity_calculator(gamma_conn_del_scz_df)





#%%


# Assuming you have four dataframes: df1, df2, df3, df4
dataframes = [delta_del_scz_df, theta_del_scz_df, alpha_del_scz_df, beta_del_scz_df,gamma_del_scz_df]

for i, df in enumerate(dataframes, 1):
    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(df, dtype=bool))
    np.fill_diagonal(mask, False)
    ax = plt.axes()
    ax.set_title(f'Heatmap {df}')
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax, vmin=0.05, vmax=0.45) 
    plt.savefig(f'heatmap_{i}.png')  # Save each plot with a unique filename
    plt.show()



#%%

# Datafram of connectivity values for each frequency band
delta_del_scz_df  = connectivity_calculator(delta_conn_del_scz_df)
theta_del_scz_df = connectivity_calculator(theta_conn_del_scz_df)
alpha_del_scz_df = connectivity_calculator(alpha_conn_del_scz_df)
beta_del_scz_df = connectivity_calculator(beta_conn_del_scz_df)
gamma_del_scz_df = connectivity_calculator(gamma_conn_del_scz_df)



dataframes = [delta_del_scz_df, theta_del_scz_df, alpha_del_scz_df, beta_del_scz_df,gamma_del_scz_df]

frequency_bands = ['delta_del_scz_df', 'theta_del_scz_df','alpha_del_scz_df','beta_del_scz_df','gamma_del_scz_df']  # Assuming the order of the dataframes corresponds to these bands

for i, (df, frequency_band) in enumerate(zip(dataframes, frequency_bands), 1):
    plot_heatmap(df, frequency_band, i)


