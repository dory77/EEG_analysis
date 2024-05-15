#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:58:29 2024

@author: anton
"""

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
import pandas as pd
import numpy as np
import os.path as op
import os
import glob
import re
from mne.preprocessing import ICA

%matplotlib qt
matplotlib.use('Qt5Agg')
print("mne version", mne.__version__)  # mne version 1.1.0


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


# %%% DEFINITION OF PATHS

input_path = "/home/monir/Monir/IPNP/22q11_analysis/28_input/Subject_28.edf"
# input_path = '/home/anton/Encfs/Private/DEMETER/22q11/data_clean/Data_22q11_edf'
# phenotype = pd.read_csv('XXXXXXXXXXXXXX')
output_path_eyes_closed = '/home/monir/Monir/IPNP/22q11_analysis/28_eyes_closed'
# output_path_clean_data_MM = '/media/anton/Elements/Catatonia/clean_data_after_visual_control'
output_path_clean_data = '/home/monir/Monir/IPNP/22q11_analysis/28_eyes_closed_clean'
# output_path_exclusion_from_MS_analysis = "/media/anton/Elements/Catatonia/clean_data_after_visual_control_artefact_multiples"
output_directory_microstates = "/home/monir/Monir/IPNP/22q11_analysis/Microstates"
# Create the directory if it doesn't exist
os.makedirs(output_directory_microstates, exist_ok=True)

montage = make_standard_montage('standard_1020')  # montage.plot()

ord_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3',
                'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']


montage.plot()
# %%% IDENTIFICATION OF ALL RELEVANT ANNOTATIONS IN ALL EEG FILES
# reading the files

event_keys = []
for eeg_file in glob.glob(input_path + '/*'):
    # print(eeg_file)
    raw = mne.io.read_raw_edf(eeg_file, encoding='latin1')
    events, event_id = mne.events_from_annotations(raw)
    event_keys += list(event_id.keys())  # len(event_keys) = 348


# A set in Python is an unordered collection of unique elements. When you convert a list to a set,
# it automatically eliminates duplicate elements, leaving only the unique ones.
event_keys = set(event_keys)  # len(event_keys) = 122
annotations_eyes_closed = {'Eyes Closed','YEUX TENUS', 'YF', 'Yeux Fermés', 'yf'}
annotations_eyes_open = {'Eyes Open', 'YO', 'Yeux Ouverts', 'yo','1 hz',
 '1 hz ',
 '10 hz',
 '13 hz',
 '15 hz',
 '18 hz',
 '20 hz',
 '25  hz',
 '3 hz',
 '5 hz',
 '5 hz ',
 '50 hz',
 '8 hz',
 'SLI: SLI: 1.0 Hz',
 'SLI: SLI: 10.0 Hz',
 'SLI: SLI: 13.0 Hz',
 'SLI: SLI: 15.0 Hz',
 'SLI: SLI: 18.0 Hz',
 'SLI: SLI: 20.0 Hz',
 'SLI: SLI: 25.0 Hz',
 'SLI: SLI: 3.0 Hz',
 'SLI: SLI: 5.0 Hz',
 'SLI: SLI: 50.0 Hz',
 'SLI: SLI: 8.0 Hz',
 'SLI=10Hz',
 'SLI=13Hz',
 'SLI=15Hz',
 'SLI=17Hz',
 'SLI=1Hz',
 'SLI=20Hz',
 'SLI=22Hz',
 'SLI=25Hz',
 'SLI=30Hz',
 'SLI=3Hz',
 'SLI=5Hz',
 'SLI=7Hz'}

eeg_file = '/home/monir/Monir/IPNP/22q11_analysis/28_input/Subject_28.edf'
raw = mne.io.read_raw_edf(eeg_file, encoding='latin1')
raw.load_data()
raw.pick_channels(ord_channels)
raw.plot_sensors()
# raw.load_data()

montage.plot()
raw.filter(0.5, 45)
raw.load_data()


events, event_id = mne.events_from_annotations(raw)
# set duration of events
events = pd.DataFrame(events, columns=["onset", "duration", "event_code"])
for i in range(len(events.onset)):
    if i != max(range(len(events.onset))):
        events.duration.iloc[i] = events.onset.iloc[i +
                                                    1] - events.onset.iloc[i]
    else:
        events.duration.iloc[i] = raw.n_times - events.onset.iloc[i]
event_id = {y: x for x, y in event_id.items()}
events.event_code = events.event_code.map(event_id)
events.duration = events.duration/raw.info["sfreq"]
events.onset = events.onset/raw.info["sfreq"]  # 256
# get limits of segments and discard one-off annotations
events = events[events.event_code.isin(annotations_eyes_closed)]
events = events.reset_index(drop=True)
list_min_max = []
for i in events.index:
    # discard one-off annotations
    if events.onset[i] != events.onset[i]+events.duration[i]:
        # -0.01 to make sure we never go beyond limit max
        list_min_max += [[events.onset[i],
                          events.onset[i]+events.duration[i]-0.01]]
# concatenate consecutive crops
tmin_0, tmax_0 = list_min_max[0]
raw_selection = raw.copy().crop(tmin=tmin_0, tmax=tmax_0)
for (tmin, tmax) in list_min_max[1:]:
    new_raw_selection = raw.copy().crop(tmin=tmin, tmax=tmax)
    raw_selection.append([new_raw_selection])

# drop non-EEG channels
if set(ord_channels).issubset(set(raw_selection.ch_names)):
    raw_selection.drop_channels(
        set(raw_selection.ch_names) - set(ord_channels))
else:
    for channel in raw_selection.ch_names:
        if channel.startswith('EEG') is False:
            # raw_selection.drop_channels(['EEG elA24'])
            raw_selection.drop_channels([channel])
    channel_renaming_dict = {name: remove_prefix(
        name, 'EEG ') for name in raw_selection.ch_names}
    raw_selection.rename_channels(channel_renaming_dict)

# change reference to the average
# Monir Made a change on_missing = 'ignore'
raw_selection.set_montage(montage, on_missing='ignore')
raw_selection.load_data()
rereferenced_raw, ref_data = mne.set_eeg_reference(raw_selection, ref_channels='average', copy=True, projection=False,
                                                   ch_type='auto', forward=None, verbose=None)

# Monir
# save data into new file
# Extract subject number from the file path
subject_number = os.path.basename(eeg_file).split('_')[1].split('.')[0]

# Construct the new filename
filename = f"{output_path_eyes_closed}/subject_{subject_number}.fif"

# Save data into the new file
rereferenced_raw.save(filename, overwrite=True)
# %%% AUTOMATIC PREPROCESSING

# glob.glob is used to get a list of file paths matching a specified pattern.
# try example with one eeg_file first
for eeg_file in glob.glob(input_path + '/*'):
    print(eeg_file)

    # %%%%%% step 1: loading the data
    # Loading the files
    raw = mne.io.read_raw_edf(eeg_file, encoding='latin1')
    raw.pick_channels(ord_channels)
    raw.plot_sensors()
    # raw.load_data()
    
    montage.plot()

    # raw.plot_psd()

# %%%%%% step 2: bandpass filtering
# Apply a bandpass filter to the EEG data. This step filters out frequencies outside the range of 0.5 to 45 Hz.
    # j'ai élargi la bande-passante des gamma de 40 à 45 Hz car la fréquence gamma à 40 Hz est importante physiologiquement
    raw.filter(0.5, 45)
    # raw.plot()
# %%%%%% step 3: selection of segments with eyes closed
# identify set of events
    events, event_id = mne.events_from_annotations(raw)
# set duration of events
    events = pd.DataFrame(events, columns=["onset", "duration", "event_code"])
    for i in range(len(events.onset)):
        if i != max(range(len(events.onset))):
            events.duration.iloc[i] = events.onset.iloc[i +
                                                        1] - events.onset.iloc[i]
        else:
            events.duration.iloc[i] = raw.n_times - events.onset.iloc[i]
    event_id = {y: x for x, y in event_id.items()}
    events.event_code = events.event_code.map(event_id)
    events.duration = events.duration/raw.info["sfreq"]
    events.onset = events.onset/raw.info["sfreq"]  # 256
# get limits of segments and discard one-off annotations
    events = events[events.event_code.isin(annotations_eyes_closed)]
    events = events.reset_index(drop=True)
    list_min_max = []
    for i in events.index:
        # discard one-off annotations
        if events.onset[i] != events.onset[i]+events.duration[i]:
            # -0.01 to make sure we never go beyond limit max
            list_min_max += [[events.onset[i],
                              events.onset[i]+events.duration[i]-0.01]]
# concatenate consecutive crops
    tmin_0, tmax_0 = list_min_max[0]
    raw_selection = raw.copy().crop(tmin=tmin_0, tmax=tmax_0)
    for (tmin, tmax) in list_min_max[1:]:
        new_raw_selection = raw.copy().crop(tmin=tmin, tmax=tmax)
        raw_selection.append([new_raw_selection])

# drop non-EEG channels
    if set(ord_channels).issubset(set(raw_selection.ch_names)):
        raw_selection.drop_channels(
            set(raw_selection.ch_names) - set(ord_channels))
    else:
        for channel in raw_selection.ch_names:
            if channel.startswith('EEG') is False:
                # raw_selection.drop_channels(['EEG elA24'])
                raw_selection.drop_channels([channel])
        channel_renaming_dict = {name: remove_prefix(
            name, 'EEG ') for name in raw_selection.ch_names}
        raw_selection.rename_channels(channel_renaming_dict)

# change reference to the average
    # Monir Made a change on_missing = 'ignore'
    raw_selection.set_montage(montage, on_missing='ignore')
    raw_selection.load_data()
    rereferenced_raw, ref_data = mne.set_eeg_reference(raw_selection, ref_channels='average', copy=True, projection=False,
                                                       ch_type='auto', forward=None, verbose=None)

# Monir
# save data into new file
    # Extract subject number from the file path
    subject_number = os.path.basename(eeg_file).split('_')[1].split('.')[0]

    # Construct the new filename
    filename = f"{output_path_eyes_closed}/subject_{subject_number}.fif"

    # Save data into the new file
    rereferenced_raw.save(filename, overwrite=True)
    # filename = output_path_eyes_closed + '/' + eeg_file[79:-4] + '.fif'
    # rereferenced_raw.save(filename, overwrite=True)

#%%% VISUAL CORRECTION
subject_num = '/home/monir/Monir/IPNP/22q11_analysis/28_eyes_closed/subject_28.fif'
subject_read = mne.io.read_raw_fif(subject_num)
subject_read.load_data()
subject_read.plot()


filename = output_path_clean_data + '/'  + 'subject_28' + '_clean' + '.fif'
subject_read.save(filename, overwrite=True)
subject_read.plot()

ModK = ModKMeans(n_clusters=7, random_state=42)
gfp_peaks = extract_gfp_peaks(subject_read)
ModK.fit(gfp_peaks, n_jobs=10)
fig = ModK.plot()
#fig.savefig('Subject_8_new', overwrite = True)
# Not very practical but each time change the subject number !
# Annotate the plot with channel names


#%%%%%% step 1: load EEG with eyes closed EEG from the output_path_eyes_closed file
filename = '/home/monir/Monir/IPNP/22q11_analysis/28_eyes_closed_clean/subject_28_clean.fif'
ec_eeg = mne.io.read_raw_fif(filename)
ec_eeg.load_data()
ec_eeg.filter(0.5, 45)
ec_eeg.plot()


#%%%%%% step 2: visualize the EEG and highlight in red (with an annotation starting with "BAD" all artefacts:
# https://www.learningeeg.com/artifacts
ec_eeg.plot()
ec_eeg.plot_psd() # this is so that you have a global vision of your spectrum of frequencies

# has been done for all fif files
fif_file_28 = '/home/monir/Monir/IPNP/22q11_analysis/28_eyes_closed_clean/subject_28_clean.fif'
ec_eeg = mne.io.read_raw_fif(fif_file_28)
ec_eeg.plot()
ec_eeg.load_data()

#%%%%%% step 6: save data in another file for visually corrected eegs
# Not very practical but each time change the subject number !
filename = output_path_clean_data + '/'  + 'subject_24' + '_clean' + '.fif'
ec_eeg.save(filename, overwrite=False)
ec_eeg.plot()
#%%%%%% step 3: interpolation of noisy channels (but I would advise against it)
# if there is only one noisy channel, you may try to interpolate it, although it may change drastically the results further on
# so better select a very short clean section than interpolate the whole EEG
# if one channel is too noisythroughout, then interpolate it
# if there is more than one channel that is constantly noisy, put this EEG in another file for exclusion
subject_read.info['bads'] = ['F4']
subject_read.interpolate_bads(reset_bads=True)
subject_read.plot()
filename = output_path_clean_data + '/'  + 'subject_20' + '_clean' + '.fif'
subject_read.save(filename, overwrite=True)
subject_read.plot()
test = ec_eeg.copy()
test.plot()
test.interpolate_bads(reset_bads=True)

#%%%%%% step 4: independent component analysis (I would prefer also to avoid it if possible)
# do not do it if you can correct the eeg by removing some segments
# use this function only if you need to get rid of a recurrent signal easy to recognize but that takes all the recording
# such as constant blinking or heartbeats
ica = ICA(n_components=19, max_iter="auto", random_state=42)
ica.fit(ec_eeg)
ica.plot_sources(ec_eeg, show_scrollbars=False) # this allows you to see all independent components to find those that look like heartbeats or blinks
ica.plot_components() # this allows you to see the distribution of each component on the scalp, also to identify heartbeat or blinking components
# use both plot_sources and plot_components in parallel
# check with https://labeling.ucsd.edu/tutorial/labels
ica.exclude = ['ICA000', 'ICA002', 'ICA003']  # indices chosen based on various plots above
reconst_ec_eeg = ec_eeg.copy() # make a copy to preserve the original eeg
ica.apply(reconst_ec_eeg) # filter based on ica
ica.plot_components()
#%%%%%% step 5: visualize microstate clusters
# check if they look like in slide 43 from this presentation: https://docs.google.com/presentation/d/1P-PJJ3COUYVyvPg2JLjbsXshTqQIM2_Nu7hngaldvWs/edit?usp=sharing


figure_dict = {}
for file_clean in glob.glob(output_path_clean_data + '/*.fif'):
    # print(file_clean)
    raw_clean = mne.io.read_raw_fif(file_clean)
    print(raw_clean)
    ModK = ModKMeans(n_clusters=7, random_state=42)
    gfp_peaks = extract_gfp_peaks(raw_clean)
    ModK.fit(gfp_peaks, n_jobs=10)

    # Extract subject number from the filename
    subject_number = re.search(r'\d+', os.path.basename(file_clean)).group()
    
    # Plot the figure and store it in the dictionary with the subject number as the key
    fig = ModK.plot()
    
    # Add the subject number to the title of the figure
    fig.suptitle(f"Subject {subject_number}", fontsize=18)
    
    figure_dict[subject_number] = fig
    # Save the figure to the specified directory
    figure_filename = os.path.join(output_directory_microstates, f"subject_{subject_number}_microstates.png")
    fig.savefig(figure_filename)


# Annotate the plot with channel names

# if they are heterogeneous, or with obvious artefacts like in the ica from the previous step, start again the correction
# if the results remains too noisy, save it in another file for noisy EEGs.

# # si les cartes sont très bruitées, sauvegarder également le fichier à part dans output_path_exclusion_from_MS_analysis
# filename2 = output_path_exclusion_from_MS_analysis + '/'  + subject + '_clean' + '.fif'
# ec_eeg.save(filename2, overwrite=True)

#%%%%%% step 1: loading the data
# Loading the files
    raw = mne.io.read_raw_edf(eeg_file,encoding='latin1')
    raw.load_data()




### fonctions utiles à garder pour la suite ###
# raw.plot_sensors(ch_type='eeg', kind='3d',show_names=True)
# raw.plot_psd_topomap()
# raw.plot_psd_topomap(bands = {'Alpha': (8, 12)})
# channel_renaming_dict = {name: remove_prefix(name, 'EEG ') for name in raw.ch_names}
# raw.rename_channels(channel_renaming_dict) # raw.ch_names
# raw.set_channel_types({"ECG":"ecg","SLI":"stim"})
# raw.set_montage(montage) # raw.plot_sensors(ch_type='eeg', kind='topomap',show_names=True)
# # un EEG a 21 canaux au lieu de 19
# for subject in pheno_MS[pheno_MS.status == 'pc']['subject_id']:
#     eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
#     raw = mne.io.read_raw_fif(eeg_file, preload=True)
#     if len(raw.ch_names) == 21:
#         print(subject)
# ec_eeg.drop_channels(['A1','A2'])

#%%% POWER SPECTRAL DENSITY ANALYSIS

pheno = pd.read_csv('/media/anton/Elements/Catatonia/phenotype/phenotype cataeggretro.csv')
pheno = pheno[['Anonymisation','statut (pc=catatonie; pnc= non catatonique)','sexe','age']]
pheno.columns = ['subject_id', 'status','sex','age']

list_subjects_microstates = []
for file in glob.glob(output_path_clean_data + '/*'):
    list_subjects_microstates += [file[67:-10]]
assert len(list_subjects_microstates) == 89

pheno_MS = pheno[pheno.subject_id.isin(list_subjects_microstates)]
assert pheno_MS[pheno_MS.status == 'pnc'].shape[0] == 53
assert pheno_MS[pheno_MS.status == 'pc'].shape[0] == 36

# plt.rcParams.update({'font.size': 12})
# fig, ax = plt.subplots()

# # for schizophrenia without catatonia
# eeg = []
# for subject in pheno_MS[pheno_MS.status == 'pnc']['subject_id']:
#     eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
#     raw = mne.io.read_raw_fif(eeg_file, preload=True)
#     epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=False)
#     evoked = epochs.average()
#     eeg.append(evoked)
        
# mean_eeg = mne.grand_average(eeg, interpolate_bads=True, drop_bads=True)
# info = mne.create_info(raw.ch_names, sfreq=256, ch_types='eeg', verbose=None)
# mean_raw_scznocat = mne.io.RawArray(mean_eeg.data, info)
# mean_raw_scznocat.plot_psd(fmin=0, fmax=45, color='blue', area_mode=None,
#                       average=True, dB=False, estimate="amplitude", ax=ax)

# # for schizophrenia with catatonia
# eeg = []
# for subject in pheno_MS[pheno_MS.status == 'pc']['subject_id']:
#     eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
#     raw = mne.io.read_raw_fif(eeg_file, preload=True)
#     epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=False)
#     evoked = epochs.average()
#     eeg.append(evoked)
        
# mean_eeg = mne.grand_average(eeg, interpolate_bads=True, drop_bads=True)
# info = mne.create_info(raw.ch_names, sfreq=256, ch_types='eeg', verbose=None)
# mean_raw_sczcat = mne.io.RawArray(mean_eeg.data, info)
# mean_raw_sczcat.plot_psd(fmin=0, fmax=45, color='red', area_mode=None,
#                       average=True, dB=False, estimate="amplitude", ax=ax) 

# plt.title('Mean spectral power', fontsize = 12)


# c'est la répartition fréquentielle de la puissance d'un signal suivant les fréquences qui le composent

psd_results = []

for eeg_file in glob.glob(output_path_clean_data + '/*'):

#%%%%%% étape 1: création d'époques de durées identiques
    eeg = mne.io.read_raw_fif(eeg_file)
    eeg.load_data()
    #Divide continuous raw data into equal-sized consecutive epochs, Creating Fixed-Length Epochs.
    epochs = mne.make_fixed_length_epochs(eeg, duration = 2, preload=True)

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
    subject_id = os.path.basename(eeg_file).split('_')[1]

    # Append results to the list
    psd_results.append([subject_id, delta_power, theta_power, alpha_power, beta_power, gamma_power])

psd_results = pd.DataFrame(psd_results, columns=['subject_id', 'delta', 'theta', 'alpha', 'beta', 'gamma'])
psd_results['subject_id'] = pd.to_numeric(psd_results['subject_id'], errors='coerce')
psd_results.sort_values(by='subject_id', inplace=True)
psd_results.set_index('subject_id', inplace=True)

# Plotting bar plots for each frequency band
psd_results[['delta', 'theta', 'alpha', 'beta', 'gamma']].plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Power in Different Frequency Bands')
plt.xlabel('Subject ID')
plt.ylabel('Power')
# Create a figure and axis
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='small')
plt.show()
save_directory = '/home/monir/Monir/IPNP/22q11_analysis/Results'
save_title = 'Power_in_Frequency_Bands_by_Subject.png'
save_path = os.path.join(save_directory, save_title)
plt.savefig(save_path, bbox_inches='tight')
print("File saved at:", save_path)

# Instead of directly indexing the filename to extract the subject ID,
# you can use the os.path.basename function to obtain the filename without the path and extension. 
#You can then split the filename based on the underscore ('_') character to get the subject ID.
# This approach makes your code more robust to changes in the file naming conventions.

#%%%%%% étape 3: on remplit de manière itérative le tableau psd_results
#     for eeg_file in glob.glob(output_path_clean_data + '/*'):
        
#         subject_name = os.path.basename(eeg_file).split('_')[1]
        
    
#     #psd_results += [[eeg_file[67:-10], delta_power, theta_power, alpha_power, beta_power, gamma_power]]

# psd_results = pd.DataFrame(psd_results, columns=['subject_id','delta','theta','alpha','beta','gamma'])
# # remarque: lorsque l'ensemble du script sera fini, on mettra toutes les analyses dans une seule boucle, pour avoir un seul tableau

#%%%%%% étape 4: comparaison entre les groupes

# merge psd_results avec le phénotype
data_final = pd.merge(pheno_MS, psd_results,on='subject_id',how='inner')

from scipy.stats import ttest_ind

psd_results = []
for var in data_final.columns[4:]:
    cata_yes = data_final[data_final.status == 'pc'][var]
    cata_no = data_final[data_final.status == 'pnc'][var]
    mean_cata_yes = cata_yes.mean()
    mean_cata_no = cata_no.mean()
    stat, pval = ttest_ind(cata_yes, cata_no)
    psd_results += [[mean_cata_yes, mean_cata_no, stat, pval]]

psd_results = pd.DataFrame(psd_results, columns=['cata_yes_mean','cata_no_mean','stat','pval'])
#            cata_yes_mean    cata_no_mean      stat          pval
# delta      0.056038         0.045615          2.310332      0.024105   # more delta in catatonics
# theta      0.023811         0.021033          1.040124      0.302196
# alpha      0.024959         0.034458         -2.546877      0.013285   # less alpha in catatonics
# beta       0.005169         0.005833         -0.843647      0.402009
# gamma      0.001032         0.000876          1.058887      0.293632

           cata_yes_mean     cata_no_mean      stat        pval
delta      0.056038          0.048313          2.041613    0.044219
theta      0.023811          0.019109          2.209523    0.029765
alpha      0.024959          0.035030         -2.962151    0.003938
beta       0.005169          0.005583         -0.608653    0.544340
gamma      0.001032          0.000905          0.939200    0.350228

#%%%%%% étape 5: utiliser topomap pour illustrer la différence entre les groupes (si elle existe)

epochs_list = []
for subject in pheno_MS['subject_id'][pheno.status == 'pnc']: 
    eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    raw.reorder_channels(ord_channels)
    epochs = mne.make_fixed_length_epochs(raw, duration = 2, preload=True)
    epochs_list += [epochs]
epochs_all_pnc = mne.concatenate_epochs(epochs_list)

epochs_list = []
for subject in pheno_MS['subject_id'][pheno.status == 'pc']: 
    eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    raw.reorder_channels(ord_channels)
    epochs = mne.make_fixed_length_epochs(raw, duration = 2, preload=True)
    epochs_list += [epochs]
epochs_all_pc = mne.concatenate_epochs(epochs_list)

plt.rcParams.update({'font.size': 12})
fig, ax1 = plt.subplots(ncols=1)
ax = epochs_all_pnc.compute_psd(fmin=0.5, fmax=45.0).plot(average=True, picks="data", exclude="bads",color='blue',axes=ax1)
ax = epochs_all_pc.compute_psd(fmin=0.5, fmax=45.0).plot(average=True, picks="data", exclude="bads",color='red',axes=ax1)
plt.title('Mean spectral power', fontsize = 12)
line1 = mlines.Line2D(range(1), range(1), color="blue", marker="_",markersize=3)
line2 = mlines.Line2D(range(1), range(1), color="red", marker="_",markersize=3)
fig.legend((line1,line2),('SCZ','SCZ+catatonia'), fontsize='small', loc='upper right')
plt.hlines(y = 0.0, xmin=7 , xmax = 13, color= 'grey')
plt.vlines(x = 7.0, ymin= -1, ymax=35, color = 'grey', linestyles='dashed')
plt.vlines(x = 13.0, ymin= -1, ymax=35, color = 'grey', linestyles='dashed')
plt.text(x=8.5,y= 0.4,s='alpha',fontdict={'size': 9})


# distributions spatiales de la densité spectrale de puissance, moyennée sur toutes les époques et bandes de fréquence
epochs_all_pnc.compute_psd(method='multitaper').plot_topomap(normalize=True, contours=0)
epochs_all_pc.compute_psd(method='multitaper').plot_topomap(normalize=True, contours=0) #, bands = {'Alpha': (8, 12)})



#%%% CONNECTIVITY ANALYSIS

##### Monir 
import mne
import glob
import os
import numpy as np
from mne.connectivity import spectral_connectivity_epochs
import matplotlib.pyplot as plt

output_path_clean_data = 'Monir/IPNP/22q11_analysis/Eyes_closed_clean/'


delta_conn_pnc, theta_conn_pnc, alpha_conn_pnc, beta_conn_pnc, gamma_conn_pnc = list(), list(), list(), list(), list()
delta_conn_pc, theta_conn_pc, alpha_conn_pc, beta_conn_pc, gamma_conn_pc = list(), list(), list(), list(), list()


for eeg_file in glob.glob(output_path_clean_data + '/*'):
    # Load raw EEG data
    eeg = mne.io.read_raw_fif(eeg_file)
    eeg.load_data()

    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)

    # Compute connectivity matrices
    conn = spectral_connectivity_epochs(epochs, method='wpli',
                                        fmin=(0.5, 4, 8, 12, 30),
                                        fmax=(4, 8, 12, 30, 45),
                                        faverage=True).get_data(output='dense')

    # Append connectivity matrices for each frequency band
    delta_conn_pc.append(conn[:,:,0])
    theta_conn_pc.append(conn[:,:,1])
    alpha_conn_pc.append(conn[:,:,2])
    beta_conn_pc.append(conn[:,:,3])
    gamma_conn_pc.append(conn[:,:,4])

# Now, you have lists containing connectivity matrices for different frequency bands
# delta_conn_pc, theta_conn_pc, alpha_conn_pc, beta_conn_pc, gamma_conn_pc

frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Plotting the connectivity matrices for each frequency band
for i, (matrix_list, band_name) in enumerate(zip([delta_conn_pc, theta_conn_pc, alpha_conn_pc, beta_conn_pc, gamma_conn_pc], frequency_bands), 1):
    plt.figure(figsize=(8, 6))
    average_matrix = np.mean(matrix_list, axis=0)  # Compute the average matrix
    sns.heatmap(average_matrix, cmap='viridis', annot=True, fmt=".2f", cbar_kws={'label': 'Connectivity'})
    plt.title(f'Connectivity Matrix ({band_name} Band)')
    plt.xlabel('Channels')
    plt.ylabel('Channels')
    plt.show()
    #plt.savefig(f'/home/monir/Monir/IPNP/22q11_analysis/Connectivity/connectivity_matrix_{band_name}.png')


delta_conn_pc = np.dstack(delta_conn_pc) 
theta_conn_pc = np.dstack(theta_conn_pc) 
alpha_conn_pc = np.dstack(alpha_conn_pc) 
beta_conn_pc = np.dstack(beta_conn_pc) 
gamma_conn_pc = np.dstack(gamma_conn_pc) 


delta_conn_pc = np.sum(delta_conn_pc, axis=2)/36
theta_conn_pc = np.sum(theta_conn_pc, axis=2)/36
alpha_conn_pc = np.sum(alpha_conn_pc, axis=2)/36
beta_conn_pc = np.sum(beta_conn_pc, axis=2)/36
gamma_conn_pc = np.sum(gamma_conn_pc, axis=2)/36

plot_sensors_connectivity(epochs.info,gamma_conn_pc)
plot_sensors_connectivity(epochs.info,beta_conn_pc)
plot_sensors_connectivity(epochs.info,alpha_conn_pc)
plot_sensors_connectivity(epochs.info,theta_conn_pc)
plot_sensors_connectivity(epochs.info,delta_conn_pc)




#%%%%%% étape 6: graphiques


#### Anton 
"""
Dans chaque bande-fréquence, y a-t-il des différences de niveau de corrélation moyen (wPLI) 
de deux électrodes entre les deux groupes ?
"""

from mne_connectivity import spectral_connectivity_epochs 
from mne_connectivity.viz import plot_sensors_connectivity
from mne.datasets import sample

elta_conn_pnc, theta_conn_pnc, alpha_conn_pnc, beta_conn_pnc, gamma_conn_pnc = list(), list(), list(), list(), list()
delta_conn_pc, theta_conn_pc, alpha_conn_pc, beta_conn_pc, gamma_conn_pc = list(), list(), list(), list(), list()

# for subject in pheno_MS['subject_id'][pheno.status == 'pnc']:
for subject in pheno_MS['subject_id'][pheno.status == 'pc']: 
    eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)

#%%%%%% étape 1: création d'époques de durées identiques

    epochs = mne.make_fixed_length_epochs(eeg, duration = 2, preload=True)

#%%%%%% étape 2: obtention d'une matrice connectivité reposant sur le weighted phase lag index pour chaque bande-fréquence

    conn = spectral_connectivity_epochs(epochs, method='wpli', # method: 'pli', 'dpli', 'wpli'
                                        fmin=(0.5,4,8,12,30), fmax=(4,8,12,30,45),
                                        faverage=True).get_data(output='dense')
    # assert conn.shape == (19, 19, 5)
    # plot_sensors_connectivity(epochs.info,conn[:, :, 2])

#%%%%%% étape 3: concaténation des matrices de corrélation pour chaque bande fréquenc

    delta_conn_pc.append(conn[:,:,0])
    theta_conn_pc.append(conn[:,:,1])
    alpha_conn_pc.append(conn[:,:,2])
    beta_conn_pc.append(conn[:,:,3])
    gamma_conn_pc.append(conn[:,:,4])    
    
 
#%%%%%% étape 4: transformation de la liste de matrices en une matrice à trois dimensions


delta_conn_pc = np.dstack(delta_conn_pc) 
theta_conn_pc = np.dstack(theta_conn_pc) 
alpha_conn_pc = np.dstack(alpha_conn_pc) 
beta_conn_pc = np.dstack(beta_conn_pc) 
gamma_conn_pc = np.dstack(gamma_conn_pc) 

#%%%%%% étape 5: moyenne par groupe


delta_conn_pc = np.sum(delta_conn_pc, axis=2)/36
theta_conn_pc = np.sum(theta_conn_pc, axis=2)/36
alpha_conn_pc = np.sum(alpha_conn_pc, axis=2)/36
beta_conn_pc = np.sum(beta_conn_pc, axis=2)/36
gamma_conn_pc = np.sum(gamma_conn_pc, axis=2)/36


#%%%%%% étape 6: graphiques


#plot_sensors_connectivity(epochs.info,delta_conn_pnc)
plot_sensors_connectivity(epochs.info,delta_conn_pc)

#plot_sensors_connectivity(epochs.info,theta_conn_pnc)
plot_sensors_connectivity(epochs.info,theta_conn_pc)

#plot_sensors_connectivity(epochs.info,alpha_conn_pnc)
plot_sensors_connectivity(epochs.info,alpha_conn_pc)

#plot_sensors_connectivity(epochs.info,beta_conn_pnc)
plot_sensors_connectivity(epochs.info,beta_conn_pc)

#plot_sensors_connectivity(epochs.info,gamma_conn_pnc)
plot_sensors_connectivity(epochs.info,gamma_conn_pc)


"""
Get wPLI for each pair of channels for each subject
Get network characteristics for each subject
Then compare groups

Note: the plot_sensors_connectivity plots only the top 20 connections - check how the threshold is set
On graph analysis, we will use all pairs
"""


plot_sensors_connectivity()


   
   # delta_conn, theta_conn, alpha_conn, beta_conn, gamma_conn = conn[:,:,0], conn[:,:,1].mean(), conn[:,:,2].mean(), conn[:,:,3].mean(), conn[:,:,4].mean()
   #  connectivity_results += [[eeg_file[64:-10], delta_conn, theta_conn, alpha_conn, beta_conn, gamma_conn]]

# connectivity_results = pd.DataFrame(connectivity_results, columns=['subject_id','delta','theta','alpha','beta','gamma'])

# data_final = pd.merge(pheno, connectivity_results,on='subject_id',how='inner')

# from scipy.stats import ttest_ind

# connectivity_results = []
# for var in data_final.columns[2:]:
#     cata_yes = data_final[data_final.statut == 'pc'][var]
#     cata_no = data_final[data_final.statut == 'pnc'][var]
#     mean_cata_yes = cata_yes.mean()
#     mean_cata_no = cata_no.mean()
#     stat, pval = ttest_ind(cata_yes, cata_no)
#     connectivity_results += [[mean_cata_yes, mean_cata_no, stat, pval]]

# connectivity_results = pd.DataFrame(connectivity_results, columns=['cata_yes_mean','cata_no_mean','stat','pval'])
# #    cata_yes_mean  cata_no_mean      stat      pval
# # 0       0.093868      0.081662  1.415992  0.160660
# # 1       0.127894      0.118503  0.978098  0.330974
# # 2       0.155953      0.168519 -0.858631  0.393109
# # 3       0.101327      0.110531 -1.157067  0.250688
# # 4       0.092825      0.089848  0.273368  0.785275

#%%%%%% étape 4: récupérer connectivité entre chaque électrode

    # faire la même chose que dans source_localization: concaténer les matrices de corrélation pour chaque bande-fréquence
    # puis construire un graphe composé des moyennes de wPLI pour l'ensemble des sujets

#%%%%%% étape 5: analyse de graphe

    # bien sélectionner les variables - ne pas perdre trop de temps
    
    
    # comparaison de graphes:  graspologic 


#%%% EXCITATION/INHIBITION RATIO

import numpy as np
import pandas as pd
import mne
import matplotlib
import pathlib
import matplotlib.pyplot as plt
import os
import os.path as op
import sys
import logging
import scipy
from scipy import signal
from scipy.signal import detrend
from scipy.stats import pearsonr
from scipy.signal import hilbert
from functools import partial
%matplotlib qt
import glob
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from mne.channels import make_standard_montage
from numpy.matlib import repmat
from PyAstronomy.pyasl import generalizedESD
import multiprocessing
from mne.filter import next_fast_len
from joblib import Parallel, delayed


def calculate_fei(Signal, window_size, window_overlap):
    
    num_channels, length_signal = Signal.shape
    window_offset = int(window_size * (1-window_overlap))
    all_window_index = create_window_indices(length_signal, window_size, window_offset)
    num_windows = all_window_index.shape[0]
    
    EI = np.zeros((num_channels,))
    EI[:] = np.NAN
    
    fEI_outliers_removed = np.zeros((num_channels,))
    fEI_outliers_removed[:] = np.NAN
    
    num_outliers = np.zeros((num_channels,num_windows))
    num_outliers[:] = np.NAN
    
    wAmp = np.zeros((num_channels, num_windows))
    wAmp[:] = np.NAN
    
    wDNF = np.zeros((num_channels, num_windows))
    wDNF[:] = np.NAN
    
  

    for i_channel in range(num_channels):
        original_amplitude = Signal[i_channel,:]  
        
        if np.min(original_amplitude) == np.max(original_amplitude):
            print('Problem computing fEI for i_channel'+str(i_channel))
            continue
        signal_profile = np.cumsum(original_amplitude - np.mean(original_amplitude))  
         
        w_original_amplitude = np.mean(original_amplitude[all_window_index], axis=1)
        xAmp = np.repeat(w_original_amplitude[:, np.newaxis], window_size, axis=1) 
        
        xSignal = signal_profile[all_window_index]                                        
        xSignal = np.divide(xSignal, xAmp).T   
                     
        _, fluc, _, _, _ = np.polyfit(np.arange(window_size), xSignal, deg=1, full=True) # arthur
        # Convert to root-mean squared error, from squared error
        w_detrended_normalized_fluctuations = np.sqrt(fluc / window_size) #arthur 
    
        EI[i_channel] = 1 - pearsonr(w_detrended_normalized_fluctuations, w_original_amplitude)[0]
        #EI[i_channel] = 1 - np.corrcoef(w_original_amplitude, w_detrended_normalized_fluctuations)[0, 1] arthur script 
        # np.corrcoef et pearsonr font exactement la même chose aussi 
        
        
        gesd_alpha = 0.05
        max_outliers_percentage = 0.025  # this is set to 0.025 per dimension (2-dim: wAmp and wDNF), so 0.05 is max
        max_num_outliers = int(np.round(max_outliers_percentage * len(w_original_amplitude)))
        outlier_indexes_wAmp = generalizedESD(w_original_amplitude, max_num_outliers, gesd_alpha)[1] #1 
        outlier_indexes_wDNF = generalizedESD(w_detrended_normalized_fluctuations, max_num_outliers, gesd_alpha)[1] #1
        outlier_union = outlier_indexes_wAmp + outlier_indexes_wDNF
        num_outliers[i_channel, :] = len(outlier_union)
        not_outlier_both = np.setdiff1d(np.arange(len(w_original_amplitude)), np.array(outlier_union))
        fEI_outliers_removed[i_channel] = 1 - np.corrcoef(w_original_amplitude[not_outlier_both], \
                                                       w_detrended_normalized_fluctuations[not_outlier_both])[0, 1]

        wAmp[i_channel,:] = w_original_amplitude
        wDNF[i_channel,:] = w_detrended_normalized_fluctuations

        EI[DFAExponent <= 0.6] = np.nan
        fEI_outliers_removed[DFAExponent <= 0.6] = np.nan
        
    return EI, fEI_outliers_removed, wAmp, wDNF


    """
    Parameters
    ----------
    signal: array, shape(n_channels,n_times) amplitude envelope for all channels
    sfreq: integer sampling frequency of the signal
    window_size: float window size (i.e 5000)
    window_overlap: float fraction of overlap between windows (0-1)
    DFAExponent: array, shape(n_channels) array of DFA values, with corresponding value for each channel, used for thresholding fEI


    Returns
    -------
    EI : array, shape(n_channels) fEI values, with wAmp and wDNF outliers 
    fEI_outliers_removed: array, shape(n_channels) fEI values, with outliers removed
    wAmp: array, shape(n_channels, num_windows) windowed amplitude, computed across all channels/windows
    wDNF: array, shape(n_channels, num_windows) windowed detrended normalized fluctuation, computed across all channels/windows
    """   


def calculate_DFA(Signal, windowSizes, windowOverlap):
    numChannels,lengthSignal = Signal.shape
    meanDF = np.zeros((numChannels, len(windowSizes)))
    DFAExponent = np.zeros((numChannels,))
    #windowSizes = windowSizes.reshape(-1, 1) 
    
    for i_channel in range(numChannels):
        for i_windowSize in range(len(windowSizes)):
            windowOffset = int(windowSizes[i_windowSize] * (1 - windowOverlap))
            allWindowIndex = create_window_indices(lengthSignal, windowSizes[i_windowSize], windowOffset)
            originalAmplitude = Signal[i_channel,:]
            signalProfile = np.cumsum(originalAmplitude - np.mean(originalAmplitude))
            xSignal = signalProfile[allWindowIndex]
            
            # Calculate local trend, as the line of best fit within the time window -> fluc is the sum of squared residuals
            _, fluc, _, _, _ = np.polyfit(np.arange(windowSizes[i_windowSize]), xSignal.T, deg=1, full=True)
            # Convert to root-mean squared error, from squared error
            det_fluc = np.sqrt(np.mean(fluc / windowSizes[i_windowSize]))
            meanDF[i_channel, i_windowSize] = det_fluc       
        
        # get the positions of the first and last window sizes used for fitting
        fit_interval_first_window = np.argwhere(windowSizes >= fit_interval[0] * sfreq)[0][0]
        fit_interval_last_window = np.argwhere(windowSizes <= fit_interval[1] * sfreq)[-1][0]

        x = np.log10(windowSizes[fit_interval_first_window:fit_interval_last_window]).reshape(-1)
        y = np.log10(meanDF[i_channel, fit_interval_first_window:fit_interval_last_window]).reshape(-1)

        model = np.polyfit(x, y, 1)
        #dfa_intercept[ch_idx] = model[1]
        DFAExponent[i_channel] = model[0]
        

    return (DFAExponent,meanDF, windowSizes)

""" 
This function calculates the Detrended Fluctuation Analysis (DFA) exponent in Python based on the provided input signal and parameters. It takes the following parameters:

Parameters: 
    Signal: 
        signal in numpy array with shape (numChannels, lengthSignal) where numChannels is the number of channels and lengthSignal is the length of the signal.
        
    windowSizes: 
        A numpy array of integers representing the window sizes used for analysis. 
    windowOverlap: 
        A float representing the overlap between consecutive windows (between 0-1).

Returns:

    DFAExponent: A numpy array of shape (numChannels,) containing the calculated DFA exponents for each channel.
    meanDF: A numpy array of shape (numChannels, numWindowSizes) containing the mean detrended fluctuations for each channel and window size.
    windowSizes: The same input windowSizes array.

"""

def create_window_indices(length_signal, length_window, window_offset):
    window_starts = np.arange(0, length_signal-length_window+1, window_offset)
    num_windows = window_starts.shape

    one_window_index = np.arange(length_window)
    all_window_index = np.repeat(one_window_index[np.newaxis, :], num_windows, axis=0)

    all_window_index += np.repeat(window_starts[:, np.newaxis], length_window, axis=1)
    return all_window_index


#%%%%% Example to use functions """
# First, define parameters 

#Specify frequency band
frequency_band = [8,13]

#fEI parameters
window_size_sec = 5
sfreq = 256
window_size = int(window_size_sec * sfreq)
window_overlap = 0.8 

#DFA parameters
windowOverlap = 0.5
compute_interval = [1,10] #interval over which to compute DFA, 
fit_interval = [1,10]
# compute DFA window sizes: 20 windows sizes per order of magnitude
windowSizes = np.floor(np.logspace(-1, 3, 81) * sfreq).astype(int)  # %logspace from 0.1 seccond (10^-1) to 1000 (10^3) seconds
# make sure there are no duplicates after rounding
windowSizes = np.sort(np.unique(windowSizes))
windowSizes = windowSizes[(windowSizes >= compute_interval[0] * sfreq) & \
                            (windowSizes <= compute_interval[1] * sfreq)]



#%%%%% Exemple to use the functions and compute fEI for several EEG
# Define the folder containing the EEG files
folder = '/media/anton/Elements/Catatonia/clean_data_after_visual_control_AI'

## Define channel names
@WARNING 
# You need to load a eeg and extract ch_names as follow : 
#ch_names=raw.info['ch_names'] 
#Or you can load a text file with ch_names 
#np.savetxt('ch_names.txt', ch_names,delimiter=',', fmt='%s') 
# ch_names= np.loadtxt('ch_names_txt_file',dtype=str)
ch_names = ord_channels


# Create df to store results 
df=pd.DataFrame()
# create list for files with bads empty = False and DFA <0.6 
bad_files = []
fEI_bad_files = []


# Loop through all EEG files in a folder
for file in glob.glob(os.path.join(folder, '*.fif')): ## modify according to your eeg format
    
    # Load the EEG file
    raw = mne.io.read_raw_fif(file, preload=True) 
    
    # Check if there are still bad channels
    if raw.info['bads']:
    # If empty= False, then Append the filename to the list of bad files
        bad_files.append(op.basename(file).split('.')[0])
        continue

    # Pre-process 
    #Filter 
    raw = raw.filter(l_freq=frequency_band[0],h_freq=frequency_band[1],
                                 filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                 fir_window='hamming',phase='zero',fir_design="firwin",
                                 pad='reflect_limited', verbose=0)

    # remove first and last second for edge effects 
    start_time = raw.times[0] + 1.0
    end_time = raw.times[-1] - 1.0
    raw=raw.crop(tmin=start_time, tmax=end_time)
    
    #Compute Hilbert tranform for amplitude envelope of the filtered signal 
    raw =raw.apply_hilbert(picks=['eeg'], envelope = True)
    
    # Select the data to use in the calculation of fEi and DFA_exponent (convert signal in numpy array)
    Signal = raw.get_data(reject_by_annotation='omit',picks='eeg')

    #compute DFA first to removed fEI_outliers with DFAExponent<0.6
    DFAExponent, meanDF, windowSizes = calculate_DFA(Signal, windowSizes, windowOverlap)
    DFA_mean = np.nanmean(DFAExponent)
    
    # Compute EI ratio 
    try:
        EI, fEI_outliers_removed, wAmp, wDNF = calculate_fei(Signal, window_size, window_overlap)
        
    except :
        fEI_bad_files.append(op.basename(file).split('.')[0])
        continue
    
    
    fEI = np.nanmean(fEI_outliers_removed)
    #else:
    #    fEI = np.nan
    
    # Add the subject ID, fEI, DFA to DF 
    subject = op.basename(file).split('.')[0]
    new_row = {'subject': subject, 
        'DFA_criterion': DFA_mean,
        'fEI': fEI, 
        **dict(zip(ch_names, fEI_outliers_removed)),
        **{f'EI_{ch}': val for ch, val in zip(ch_names, EI)},}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

#%%%%% Group comparison

pheno_MS = pheno_MS.rename(columns={'subject_id':'subject'})
df_final = df.copy()
for i in range(len(df_final.subject)):
    df_final.subject.iloc[i] = df_final.subject.iloc[i].removesuffix('_clean')   
df_final = pd.merge(pheno_MS, df_final, on='subject',how='inner')

import dabest

f, axx = plt.subplots(nrows=1, ncols=2,
                        figsize=(50, 30),
                        gridspec_kw={'wspace': 0.25} # ensure proper width-wise spacing.
                       )
i = 0

for col in ['fEI','DFA_criterion']:
col = 'fEI'
    non_cat = df_final[col][df_final.status == 'pnc']
    cat = df_final[col][df_final.status == 'pc']
    
    res = pd.concat([non_cat, cat], axis=1)
    res.columns = ["noCAT","CAT"]
    
    res = dabest.load(res, idx=("noCAT","CAT"))
    # res_stats = res.cohens_d.statistical_tests
    # res_stats.to_csv('/home/anton/Encfs/Private/DEMETER/EEG_RDB/Results/Table2_study1_stats_C_R_A_%s' % col) 
    # res_stats.to_csv('/home/anton/Encfs/Private/DEMETER/Retrospective_study_on_patients_2018_2019_2020/Figures_and_results/Analysis_with_four_clusters/Table3_stats_class_D_%s' % col)
    #res_stats.to_csv('/home/anton/Encfs/Private/DEMETER/Retrospective_study_on_patients_2018_2019_2020/Figures_and_results/Analysis_with_four_clusters/Table3_stats_class_C_%s' % col)
    #res.mean_diff.plot(ax=axx.flat[i])
    # res.cliffs_delta.plot(ax=axx.flat[i],raw_marker_size=3)
    res.cohens_d.plot()
        ax=axx.flat[i],raw_marker_size=3)
    i+=1
    
from scipy.stats import ttest_ind
ttest_ind(cat,non_cat)

#%%%%% Exemple to compute fEI for single EEG
# Load eeg 
filename = '/Path/to/eeg_file'
raw = mne.io.read_raw_fif(filename, preload=(True),verbose=True) 

#Drop bad channels 
raw= raw.drop_channels(['E14','E17','E21','E48','E119','E126','E127'])
raw.info['bads'] # check that there are no more bads

#""" Filtered between 8 and 13 Hz """
raw = raw.filter(l_freq=frequency_band[0],h_freq=frequency_band[1],
                             filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                             fir_window='hamming',phase='zero',fir_design="firwin",
                             pad='reflect_limited', verbose=0)



# remove first and last second for edge effects 
start_time = raw.times[0] +1
end_time = raw.times[-1] - 1
raw=raw.crop(tmin=start_time, tmax=end_time)

# Compute Hilbert amplitude envelope of the filtered signal
raw=raw.apply_hilbert(picks=['eeg'], envelope = True,verbose='debug')


## Extract data in numpy array
Signal = raw.get_data(reject_by_annotation='omit', picks = 'eeg')

# Compute DFA and fEI 
DFAExponent, meanDF, windowSizes = calculate_DFA(Signal, windowSizes, windowOverlap)
DFA_criterion = np.mean(DFAExponent)
EI,fEI_outliers_removed, wAmp, wDNF = calculate_fei(Signal, window_size, window_overlap)


#%%%%% Example to plot topomap with fEI for each electrode

# First, extract x, y coordinates 
#EXTRACT x,y coordinates for each electrodes that fit to 2D topomap : Plot sensors then extract coordinates from the figure
# you need to load one EEG
fig = raw.plot_sensors(show_names=True) ## also works with fig=mne.viz.plot_montage(your_montage)
ax = fig.axes[0]
coordinates = []
for point in ax.collections[0].get_offsets():
    x, y = point
    coordinates.append([x, y])
xy_pos= np.array(coordinates)
np.savetxt('electrode_positions.txt', xy_pos) # Save coordinates in txt as references coordinates

xy_pos= np.loadtxt('electrode_positions.txt')

# determine colormap/colorbar intervall as Bruining et al 2020 
def plot_interval(values):
    # Determine the minimum and maximum of the range
    min_range = np.nanpercentile(values, 5)
    max_range = np.nanpercentile(values, 95)

    # Make the interval symmetric around 1
    range_diff = max(1 - min_range, max_range - 1)
    min_range_s = 1 - range_diff
    max_range_s = 1 + range_diff

    # Round the range to one decimal place
    min_range_r = np.round(min_range_s, 1)
    max_range_r = np.round(max_range_s, 1)

    return   min_range_r, max_range_r

list(df_final)
ei_all = df_final.iloc[:, 6:25][df_final.status == 'pc'] ## to obtain the ei ratio for each electrodes for all subject
ei_all= np.nanmean(ei_all,axis=0)
min_range_r, max_range_r = plot_interval(ei_all)

#topomap 
#WARNING : If the plot not displayed, load one record, try raw.plot() then retry
ig, ax = plt.subplots()
im, _ = mne.viz.plot_topomap(ei_all, pos=xy_pos, vlim=(min_range_r, max_range_r),  cmap='bwr'  ,contours=0,axes=ax)
cbar = plt.colorbar(im, ax=ax)
plt.legend(loc='lower center')
plt.gcf().set_size_inches(7, 6)
plt.subplots_adjust(top=0.94,
bottom=0.048,
left=0.053,
right=0.985,
hspace=0.2,
wspace=0.2)
ax.set_title('Excitation/Inhibition ratio in SCZ cat', fontsize= 12, fontweight='bold')
plt.show()


#%%% MICROSTATES ANALYSIS

print(pd. __version__)

# merge psd_results avec le phénotype
pheno = pd.read_csv('/media/anton/Elements/Catatonia/phenotype/phenotype cataeggretro.csv')
pheno = pheno[['Anonymisation','statut (pc=catatonie; pnc= non catatonique)','sexe','age']]
pheno.columns = ['subject_id', 'status','sex','age']

list_subjects_microstates = []
for file in glob.glob(output_path_clean_data + '/*'):
    list_subjects_microstates += [file[67:-10]]
assert len(list_subjects_microstates) == 66

pheno_MS = pheno[pheno.subject_id.isin(list_subjects_microstates)]
assert pheno_MS[pheno_MS.status == 'pnc'].shape[0] == 30
assert pheno_MS[pheno_MS.status == 'pc'].shape[0] == 36
    
#%%%%%% 4.1) find the optimal number of clusters

n_jobs = 10
cluster_numbers = range(2,9)
scores = {"Silhouette": np.zeros(len(cluster_numbers)),
    "Calinski-Harabasaz": np.zeros(len(cluster_numbers)),
    "Dunn": np.zeros(len(cluster_numbers)),
    "Davies-Bouldin": np.zeros(len(cluster_numbers))}

for k, n_clusters in enumerate(cluster_numbers):
   
    # vary the number of clusters
    ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
            
    individual_gfp_peaks = list()
    for subject in pheno_MS['subject_id']:
    # for subject in pheno_MS['subject_id'][pheno.status == 'pc']: 
        eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
        raw = mne.io.read_raw_fif(eeg_file, preload=True)
        raw.reorder_channels(ord_channels)
        # Extract Gfp peaks
        gfp_peaks = extract_gfp_peaks(raw) #, return_all=True)
        # equalize peak number across subjects by resampling
        gfp_peaks = resample(gfp_peaks, n_resamples=1, n_samples=200, random_state=42)[0]
        individual_gfp_peaks.append(gfp_peaks.get_data())

    individual_gfp_peaks = np.hstack(individual_gfp_peaks)    
    individual_gfp_peaks = ChData(individual_gfp_peaks, raw.info)
    
    # group level clustering
    ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
    ModK.fit(individual_gfp_peaks, n_jobs=2)
    
    scores["Silhouette"][k] = silhouette_score(ModK)
    scores["Calinski-Harabasaz"][k] = calinski_harabasz_score(ModK)
    scores["Dunn"][k] = dunn_score(ModK)
    scores["Davies-Bouldin"][k] = davies_bouldin_score(ModK) 


f, ax = plt.subplots(2, 2, sharex=True)
for k, (score, values) in enumerate(scores.items()):
    ax[k // 2, k % 2].bar(x=cluster_numbers, height=values)
    ax[k // 2, k % 2].set_title(score)
plt.text(
    0.03, 0.5, "Score",
    horizontalalignment='center',
    verticalalignment='center',
    rotation=90,
    fontdict=dict(size=14),
    transform=f.transFigure,
)
plt.text(
    0.5, 0.03, "Number of clusters",
    horizontalalignment='center',
    verticalalignment='center',
    fontdict=dict(size=14),
    transform=f.transFigure,
)
plt.show()

# invert davies-bouldin scores
scores["Davies-Bouldin"] = 1 / (1 + scores["Davies-Bouldin"])

# normalize scores using sklearn
from sklearn.preprocessing import normalize
scores = {score: normalize(value[:, np.newaxis], axis=0).ravel()
          for score, value in scores.items()}

# set width of a bar and define colors
barWidth = 0.2
colors = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F"]

# create figure
plt.figure(figsize=(10, 8))
# create the position of the bars on the X-axis
x = [[elt + k * barWidth for elt in np.arange(len(cluster_numbers))]
     for k in range(len(scores))]
# create plots
for k, (score, values) in enumerate(scores.items()):
    plt.bar(
        x=x[k],
        height=values,
        width=barWidth,
        edgecolor="grey",
        color=colors[k],
        label=score,
    )
# add labels and legend
plt.xlabel("Number of clusters")
plt.ylabel("Score normalize to unit norm")
plt.xticks(
    [pos + 1.5 * barWidth for pos in range(len(cluster_numbers))],
    [str(k) for k in cluster_numbers],
)
plt.legend()
plt.show()


#%%%%%% 4.2) set microstate topographies
n_jobs = 10

# two-step clustering
individual_gfp_peaks = list()
for subject in pheno_MS['subject_id']: #[pheno.status == 'pnc']:
    eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
    raw = mne.io.read_raw_fif(eeg_file, preload=True) 
    raw.reorder_channels(ord_channels)   
    # Extract Gfp peaks
    gfp_peaks = extract_gfp_peaks(raw)
    # equalize peak number across subjects by resampling
    gfp_peaks = resample(gfp_peaks, n_resamples=1, n_samples=200, random_state=42)[0]
    individual_gfp_peaks.append(gfp_peaks.get_data())

individual_gfp_peaks = np.hstack(individual_gfp_peaks)    
individual_gfp_peaks = ChData(individual_gfp_peaks, raw.info)

# group level clustering for scz no cat
ModK = ModKMeans(n_clusters=7, random_state=42)  #why we have seven clusters?
ModK.fit(individual_gfp_peaks, n_jobs=10)
ModK.plot()
ModK.GEV_ # 0.56 (2) 0.61 (3) 0.65 (4) 0.67 (5) 0.69 (6) 0.70 (7) 0.71 (8)
ModK.reorder_clusters(order=[5,2,0,1,3,4,6])
ModK.rename_clusters(new_names=["A","B","C","D","E","F","G"])
ModK.invert_polarity([True,False,True,False,False,False,False])
ModK.plot()


#%%%%%% 4.3) backfitting

ms_data = list()
for subject in pheno_MS['subject_id']:
    eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
    raw = mne.io.read_raw_fif(eeg_file, preload=True)  
    raw.reorder_channels(ord_channels)
    segmentation = ModK.predict(raw, factor=10, half_window_size=8)
    d = segmentation.compute_parameters()
    d["subject_id"] = subject
    ms_data.append(d)

ms_data = pd.DataFrame(ms_data)
ms_data.to_csv('/home/anton/Encfs/Private/DEMETER/Catatonia_study/results_89subjects/microstates_summary.csv',index=False)


#%%%%%% 4.4) group comparison

ms_data = pd.read_csv('/home/anton/Encfs/Private/DEMETER/Catatonia_study/results_89subjects/microstates_summary.csv')
data = pd.merge(pheno_MS,ms_data,on='subject_id',how='inner')
list(data)

table_demo = []
for value in ["pnc", "pc"]:
    df1 = data[data.status == value]
    N_tot = len(df1)
    age_mean, age_std = round(df1.age.mean(),2),round(df1.age.std(),2)
    nb_F = len(df1[df1.sex == 1])
    nb_M = len(df1[df1.sex == 0])
    table_demo += [[value, N_tot, age_mean, age_std, nb_F, nb_M]]
table_demo = pd.DataFrame(table_demo, columns=[value, N_tot, age_mean, age_std, nb_F, nb_M])
table_demo.columns = ['group','N','age_mean','age_std','nb_F','nb_M']
#     group    N  age_mean  age_std  nb_F  nb_M
# 0  no ASD   33     12.36     2.90    14    19
# 1     ASD  114     10.70     2.87    17    97
        


# statistiques
from scipy.stats import chi2_contingency, ttest_ind

list(data_young)

dfcat = data[data.status == 'pc']['age'].dropna()
dfnocat = data[data.status == 'pnc']['age'].dropna()
ttest_ind(dfcat, dfnocat) # Ttest_indResult(statistic=-2.918183379421798, pvalue=0.004082483585111544)


df_repart = np.array([[16,20],[10,20]])
chi2, p, dof, ex = chi2_contingency(df_repart, correction=True)

# compute recording EEG total times for each group

list_duration_recordings = []
for subject in pheno_MS['subject_id'][pheno_MS.status == 'pnc']:
    eeg_file = output_path_clean_data + '/'  + subject + '_clean.fif'
    raw = mne.io.read_raw_fif(eeg_file, preload=True) 
    raw.times.max()
    list_duration_recordings += [raw.times.max()]
# pnc
np.array(list_duration_recordings).mean() 548.7022135416667/60 # 9.1
np.array(list_duration_recordings).std() 126.60009750387901/60 # 2.1
# pc
pc = np.array(list_duration_recordings).mean() 549.5785303819443/60 # 9.2
np.array(list_duration_recordings).std() 198.08187714542206/60 # 3.3


ttest_ind(dfcat, dfnocat)

results = pd.merge(results, record_time, how='inner', on='subject_id')  



import pingouin as pg
import dabest

f, axx = plt.subplots(nrows=2, ncols=7,
                        figsize=(60, 30),
                        gridspec_kw={'wspace': 0.5} # ensure proper width-wise spacing.
                       )
i = 0
# for col in ['A_occurrences','B_occurrences', 'C_occurrences','D_occurrences','E_occurrences','F_occurrences','G_occurrences',
#             'A_timecov','B_timecov','C_timecov','D_timecov','E_timecov','F_timecov','G_timecov']:

for col in ['A_meandurs','B_meandurs', 'C_meandurs','D_meandurs','E_meandurs','F_meandurs','G_meandurs',
            'A_gev','B_gev','C_gev','D_gev','E_gev','F_gev','G_gev']:
    

    non_cat = data[col][data.status == 'pnc']
    cat = data[col][data.status == 'pc']
    
    res = pd.concat([non_cat, cat], axis=1)
    res.columns = ["noCAT","CAT"]
    
    res = dabest.load(res, idx=("noCAT","CAT"))
    res_stats = res.cohens_d.statistical_tests
    res_stats.to_csv('/home/anton/Encfs/Private/DEMETER/Catatonia_study/results_89subjects/ms_stats_%s' % col) 
    res.cohens_d.plot(ax=axx.flat[i],raw_marker_size=3)
    i+=1
















