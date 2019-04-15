import mne
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mne.preprocessing import eog
from mne.preprocessing import create_eog_epochs
import Tkinter
import os
import sys
sys.path.append(os.getcwd() + '/helper')
import extract_nslog_event
import scipy_eog

###########################################
#       Setup and Basic Preprocessing     #
###########################################
# specify sample subject data directory
raw_fname   = '/Users/Jin/Documents/MATLAB/research/mne-egi/data/sfv_eeg_011ts.raw'
ns_eventlog = '/Users/Jin/Documents/MATLAB/research/mne-egi/data/sfv_eeg_011ts_nsevent'

# specify sub-sample of channels to look in detail
# note that 'EB' is a newly created channel name for 'eye blink' using bipolar reference
selection = ['EB','E11','E24','E124','E36','E104','E52','E62','E92']

# let's read in the raw file
# you can specify montage (see MNE website for available montages)
# we set `preload=True` because some of the preprocessing functions require raw file to be preloaded
print('reading raw file...')
raw = mne.io.read_raw_egi(raw_fname, montage='GSN-HydroCel-128', preload=True)
print('Done!')

# in HydroCel GSN model caps:
# eye blink channels are (right, left): E8, E126, E25, E127
# horizontal eye mvmt channels are: E125, E128
# when reading in raw file, you can specify the eye channels by `eog=[]` but we will do this later at bipolar reference stage
# raw = mne.io.read_raw_egi(raw_fname, montage='GSN-HydroCel-128', eog=['E8', 'E126', 'E25', 'E127'], preload=True)
# you can check by raw.plot_sensors()

# apply bandpass filter to raw file(highpass, lowpass)
raw.filter(1,30)
# show raw summary
raw.info

###########################################
#    Creating Dataframe and Annotations   #
###########################################
# the below codes use custom codes created specifically for the experiment
# create pandas data frames for different tasks
nsdata, df_lst, df_plst, df_tlst, df_slst = extract_nslog_event.create_df(ns_eventlog)

# you can see how the data is cleaned from the events-exported text file by:
# show sample line of event of 10th item
nsdata[10]
# show data frame structure of 3rd index
df_tlst.iloc[3]

# create onset-only data frame (event tag specifications)
df_tlstS = extract_nslog_event.create_df_onset(df_tlst)
# show total events of interest
len(df_tlstS)

# find impedance onsets
imp_onset, imp_offset, imp_dur = extract_nslog_event.find_impedances(nsdata)

# annotate on raw with 'bad' tags
# params `reject_by_annotation` will search for 'bad' tags later
annot_imp = mne.Annotations(imp_onset, imp_dur, ["bad imp"] * len(imp_onset), orig_time=raw.info['meas_date'])
raw.set_annotations(annot_imp)

# let's manually mark potential bad channels
# raw.plot will show the actual raw file with annotations marked as red segments from above
# you can inspect for good/bad channels and manually click on bad channels to mark them bad
# once you manually inspected channels (colored as red), type `raw.info['bads']` to see that the channel is marked bad
# raw.plot(bad_color='red', block=True)

# or if you already know/or want to skip the plot part, you can specify more directly
raw.info['bads'] = ['E128', 'E127', 'E107', 'E56', 'E57', 'E18', 'E49', 'E48', 'E115', 'E113', 'E122', 'E121', 'E123', 'E124', 'E108', 'E63', 'E1', 'E32', 'E33']


###########################################
#     Eye-Related Artifact Detections     #
###########################################
# let's begin eye artifact detections
print('Starting EOG artifact detection')
# specify the eye channels (here I used the right side of eye channels)
# `set_bipolar_reference` will use the two channels 'E8' and 'E126' to create a virtual eye channel 'EB'
# it is basically a subtraction between the two
raw = mne.set_bipolar_reference(raw, ['E8'],['E126'],['EB'])
# specify this as the eye channel
# note that when you specify none-eog specific channels as 'eog' channels, these channels will potentially
# contain partial scalp data of frontal brain/etc
raw.set_channel_types({'EB': 'eog'})

# we have the option to use mne built-in function to find peaks or use custom built eog function using scipy
# both result in similar eye blink detections
events_eog = eog.find_eog_events(raw, reject_by_annotation=True, thresh=0.0001, verbose=None)
# raw = scipy_annotate_eyeblinks(raw, 'EB', 100)

# `events_eog` above will give where the eye blinks occured in samples
# we will convert the sample number to seconds so we can annotate on the raw file

# get just the sample numbers from the eog events
eog_sampleN = [i[0] for i in events_eog]
# convert to seconds for annotation-friendly purposes
for i in range(0, len(eog_sampleN)):
    eog_sampleN[i] = eog_sampleN[i] / float(200)

annot_eog = mne.Annotations(eog_sampleN, [0.1] * len(eog_sampleN), ["bad eye"] * len(eog_sampleN), orig_time = raw.info['meas_date'])

# add this eye blink annotation to the previous annotation by simply adding
new_annot = annot_imp + annot_eog
raw.set_annotations(new_annot)
print('new annotation set!')
# you can check that more red segments are marked on the raw file
# raw.plot(bad_color='red')

# let's set eeg reference
# now that bad channels are marked and we know which bad segments to avoid, we set eeg reference
print('setting eeg reference...')
raw.set_eeg_reference('average', projection=True)


###########################################
#            Creating Epochs              #
###########################################
# update event ids in mne events array and double check sampling onset timing as sanity check
events_tlst = mne.find_events(raw, stim_channel='tlst')
# events_tlst is a array structure ie.  (1, 0, 1) and so far, the all the event tags are 1
# which is not true. We will update the event tags with 1s and 2s with custom built function
events_tlstS = extract_nslog_event.assign_event_id(df_tlst, events_tlst)

# epoching initially with metadata applied
event_id_tlst = dict(lstS=1)
tmin = -0.25  # start of each epoch
tmax = 0.8  # end of each epoch
# set baseline to 0
baseline = (tmin, 0)

# picks specify which channels we are interested
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads')

# let's create epochs now!
# epochs_tlstS is created from the raw file with event labels of interest
# `metadata` field is used to put in our comprehensive pandas dataframe
# it is useful for later creating evoked responses by conditions
epochs_tlstS = mne.Epochs(raw, events_tlstS, event_id_tlst, tmin, tmax, proj=False, picks=picks, baseline=baseline, preload=True, reject=None, reject_by_annotation=True, metadata=df_tlstS)
print('epochs_tlstS:')
print(epochs_tlstS)

# show drop percentage from mne.Epochs
drop_count = 0
for j in range(0, len(epochs_tlstS.drop_log)):
    if 'bad eye' in epochs_tlstS.drop_log[j]:
        drop_count += 1
print(str(drop_count) + ' epochs dropped by eog annotation')
print('perecentage dropped: ' + str(epochs_tlstS.drop_log_stats()))

# create evoked respone using pandas query based on metadata created from previous epochs
evoked_tlst_c1 = epochs_tlstS["label=='lstS' and cond=='1'"].average()
evoked_tlst_c2 = epochs_tlstS["label=='lstS' and cond=='2'"].average()
evoked_tlst_c3 = epochs_tlstS["label=='lstS' and cond=='3'"].average()
evoked_tlst_c4 = epochs_tlstS["label=='lstS' and cond=='4'"].average()


###########################################
#      Advanced Artifact Detection        #
###########################################
# one of the collaborators of MNE developed auto artifact detection
# which automatically attempts to find bad channels and interpolate based on nearby channels
# we will use this and ICA to do more cleaning
# for more detail see: http://autoreject.github.io

# let's first apply independent component analysis (ICA) on the epochs to filter out bad ICs
from autoreject import get_rejection_threshold
# the function calculates for optimal reject threshold for ICA
reject = get_rejection_threshold(epochs_tlstS)

from mne.preprocessing import ICA
# For simplicity/time sake, we will specify n_components as 20
# ICA can create up to as many electrodes you have
ica = ICA(n_components=20, max_pca_components=None, n_pca_components=None, noise_cov=None, random_state=None, method='fastica', fit_params=None, max_iter=200, verbose=None)
print('fitting ica...')
ica.fit(epochs_tlstS, reject=reject)

# inspect by ICA correlation
# the mne-built in function can suggest what bad components are using the eog channels
eog_inds, scores = ica.find_bads_eog(epochs_tlstS)
print('suggested eog component: ' + str(eog_inds))
ica.plot_scores(scores, exclude=eog_inds, labels='eog')
# ica.plot_properties(raw, picks=eog_inds, psd_args={'fmax': 35.}, image_args={'sigma': 1.})

# the line below will exclude the ones suggested bad
ica.exclude += eog_inds
# you can also manually inspect each component
# the plot will prioritze to show components with greater variances
# so it is more likely to find bad components in low-numbered components
ica.plot_components(inst=epochs_tlstS)

# ica.plot_sources(inst=epochs_tlstS)
ica.apply(epochs_tlstS)
print('number of ICs dropped: ' + str(len(ica.exclude)))


# now that we have ICA applied to our epochs, let's try using autoreject cleaning
from autoreject import AutoReject
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs_tlstS)

# you can manually check the differences
epochs_clean.plot()

# now let's create a new evoked responses (ie. the autoreject evoked)
arevoked_tlst_c1 = epochs_clean["label=='lstS' and cond=='1'"].average()
arevoked_tlst_c2 = epochs_clean["label=='lstS' and cond=='2'"].average()
arevoked_tlst_c3 = epochs_clean["label=='lstS' and cond=='3'"].average()
arevoked_tlst_c4 = epochs_clean["label=='lstS' and cond=='4'"].average()


###########################################
#   Plotting Event-Related Potentials     #
###########################################
picks_select = mne.pick_types(epochs_clean.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads', selection=selection)

evoked_dict = {'highcosval': arevoked_tlst_c1,
                'lowcosval': arevoked_tlst_c2,
                'highcosinval': arevoked_tlst_c3,
                'lowcosinval': arevoked_tlst_c4}

picks_select = mne.pick_types(arevoked_tlst_c1.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads', selection=selection)


# this will plot each selected channel with comparison of two conditions
title = '%s_vs_%s_E%s.png'
for i in range(0, len(picks_select)):
    fig1 = mne.viz.plot_compare_evokeds({'highcos/val':evoked_dict['highcosval'], 'lowcos/val':evoked_dict['lowcosval']}, picks=picks_select[i], show=False)
    fig2 = mne.viz.plot_compare_evokeds({'highcos/inval':evoked_dict['highcosinval'], 'lowcos/inval':evoked_dict['lowcosinval']}, picks=picks_select[i], show=False)
    fig3 = mne.viz.plot_compare_evokeds({'highcos/val':evoked_dict['highcosval'], 'highcos/inval':evoked_dict['highcosinval']},picks=picks_select[i], show=False)
    fig4 = mne.viz.plot_compare_evokeds({'lowcos/val':evoked_dict['lowcosval'],'lowcos/inval':evoked_dict['lowcosinval']}, picks=picks_select[i], show=False)

    # save figs
    fig1.savefig(title % (evoked_dict.keys()[0], evoked_dict.keys()[1], i))
    fig2.savefig(title % (evoked_dict.keys()[2], evoked_dict.keys()[3], i))
    fig3.savefig(title % (evoked_dict.keys()[0], evoked_dict.keys()[2], i))
    fig4.savefig(title % (evoked_dict.keys()[1], evoked_dict.keys()[3], i))


# this will plot just the evoked responses per conditions with all channels
fig5 = arevoked_tlst_c1.plot(titles='cond1: high cos/val', show=False)
fig6 = arevoked_tlst_c2.plot(titles='cond2: low cos/val', show=False)
fig7 = arevoked_tlst_c3.plot(titles='cond3: high cos/inval', show=False)
fig8 = arevoked_tlst_c4.plot(titles='cond4: low cos/inval', show=False)

# save figs
fig5.savefig('c1all.png')
fig6.savefig('c2all.png')
fig7.savefig('c3all.png')
fig8.savefig('c4all.png')
