import mne
from scipy.signal import find_peaks
from scipy.signal import peak_prominences

# raw_fname   = '/Users/Jin/Documents/MATLAB/research/mne/sfv_eeg_011ts.raw'
# ns_eventlog = '/Users/Jin/Documents/MATLAB/research/mne/sfv_eeg_011ts_nsevent'

def scipy_annotate_eyeblinks(raw, eye_channel='EB', min_dist=100):
    """
    ***raw needs to be bipolar-referenced/or have specific eye channels
    finds local maximum peak points in the eye_channel and annotates on the raw file

    Parameters (defaults)
    ---------------------
    raw: Raw object
        instance of raw file
    eye_channel: str ('EB')
        eye channel in string. Can be a virtual eye channel created from
        mne.set_bipolar_reference
    min_dist: int (100)
        minimum distance to search for another local maximum
        peaks found within the next100 m/s range won't be considered peaks.
        Use this to avoid finding excessive amount of peaks

    Returns
    -------
    raw: Raw object
        the annotated raw file with local maximum peaks
    """

    data, times = raw[raw.ch_names.index(eye_channel)]
    # ie. data: array([[ 7.31836466e-19, -1.67314226e-07,  2.18515609e-07, ...,
        # -1.38256900e-06, -7.66754021e-07, -2.03287907e-19]])

    # peaks is the sample number of each peak and val creates dict of height vals
    peaks, val = find_peaks(data[0], height=0, distance=100)
    # ie. peaks
    # array([      3,    1128,    1149, ..., 1028675, 1028693, 1028710])
    # val
    # {'peak_heights': array([9.50663807e-07, 1.42416676e-04, 4.91292689e-06, ...,
       # 1.13145838e-06, 1.13396820e-06, 3.48277682e-06])}

    # get the index number for vals that exceed the threshold
    threshold_peak_indx = []
    for i in range(0, len(val['peak_heights'])):
        if val['peak_heights'][i] > 0.0001:
            threshold_peak_indx.append(i)

    # get the corresponding sample values from peaks
    eb_samples = []
    test_samp = []
    val_samp = []
    for i in threshold_peak_indx:
        eb_samples.append(peaks[i])
        test_samp.append(peaks[i])
        val_samp.append(val['peak_heights'][i])

    # convert to seconds
    for i in range(0, len(eb_samples)):
        eb_samples[i] = eb_samples[i]/float(200)

    # filter out eye blinks marked in impedance periods
    eb_samples_filtered = []
    eb_samples_filtered_test = []
    val_samp_filtered = []
    for i in range(0, len(eb_samples)):
        in_range = False
        for j in range(0, len(imp_onset)):
            if (imp_onset[j] <= eb_samples[i] <= imp_offset[j]):
                in_range = True
        if not in_range:
            eb_samples_filtered.append(eb_samples[i])
            eb_samples_filtered_test.append(test_samp[i])
            val_samp_filtered.append(val_samp[i])

    annot_eb = mne.Annotations(eb_samples_filtered, [0.1] * len(eb_samples_filtered), ["bad eye"] * len(eb_samples_filtered), orig_time = raw.info['meas_date'])
    raw.set_annotations(annot_eb)
    print('new annotation set!')
    return raw
