import pandas as pd
import numpy as np
import os


def create_df(ns_eventlog):
    """
    Create dataframes used for checking impedance periods and epochs metadata
    First creates all event df (df_lst), then creates sub dataframes

    Parameters
    ----------
    ns_eventlog: must be a string with netstation event log file name

    Returns
    -------
    currently 3 pandas dataframes (df_lst, df_plst, df_tlst, df_slst)
    """
    # values below currently fixed for sfv exp
    expected_plst = 10
    expected_tlst = 800
    expected_slst = 200

    # get the subject number
    temp = os.path.basename(ns_eventlog)
    temp = [int(s) for s in temp if s.isdigit()]
    temp = ''.join(str(x) for x in temp)

    print('creating data frame from ns event log...')
    with open(ns_eventlog) as fp:
        nsdata = fp.readlines()
    nsdata = [i.split('\t') for i in nsdata]
    #remove random '_' in front of time format
    for i in range(0,len(nsdata)):
        try:
            nsdata[i][4]=nsdata[i][4][1:]
        except IndexError:
            pass
    #convert from string of HH:MM:SS:MMM to MS
    for i in range(0,len(nsdata)):
        try:
            hh=int(nsdata[i][4][0:2]) * 3600000
            mm=int(nsdata[i][4][3:5]) * 60000
            ss=int(nsdata[i][4][6:8]) * 1000
            ms=int(nsdata[i][4][9:12])
            nsdata[i][4]=hh+mm+ss+ms
        except IndexError:
            pass
        except ValueError:
            pass

    # create data frame for labels with particular interest
    col_names = ['code','label','onset','cond','indx']
    df_lst = pd.DataFrame(columns=col_names)
    temp_code = []
    temp_label = []
    temp_onset = []
    temp_cond = []
    temp_indx = []
    for i in range(0, len(nsdata)):
        try:
            if nsdata[i][0][1:] == 'lst':
                temp_code.append(nsdata[i][0])
                temp_label.append(nsdata[i][1])
                temp_onset.append(nsdata[i][4])
                temp_cond.append(nsdata[i][7])
                temp_indx.append(nsdata[i][9])
        except IndexError:
            pass
    df_lst['code']=temp_code
    df_lst['label']=temp_label
    df_lst['onset']=temp_onset
    df_lst['cond']=temp_cond
    df_lst['indx']=temp_indx
    assert len(df_lst) == expected_slst+expected_tlst+expected_plst, "df_lst number != expected total lst"

    # create dataframe particular for plst
    df_plst = pd.DataFrame(columns=col_names)
    temp_code = []
    temp_label = []
    temp_onset = []
    temp_cond = []
    temp_indx = []
    for i in range(0, len(df_lst)):
        if df_lst.iloc[i]['code'] == 'plst':
            temp_code.append(df_lst.iloc[i]['code'])
            temp_label.append(df_lst.iloc[i]['label'])
            temp_onset.append(df_lst.iloc[i]['onset'])
            temp_cond.append(df_lst.iloc[i]['cond'])
            temp_indx.append(df_lst.iloc[i]['indx'])
    df_plst['code']=temp_code
    df_plst['label']=temp_label
    df_plst['onset']=temp_onset
    df_plst['cond']=temp_cond
    df_plst['indx']=temp_indx
    assert len(df_plst) == expected_plst, "ERROR: len(df_plst) != expected plst"
    # assert len(df_plst) == len(events_plst), "ERROR: len(df_plst) != len(events_plst)"

    # create dataframe particular for tlst
    df_tlst = pd.DataFrame(columns=col_names)
    temp_code = []
    temp_label = []
    temp_onset = []
    temp_cond = []
    temp_indx = []
    for i in range(0, len(df_lst)):
        if df_lst.iloc[i]['code'] == 'tlst':
            temp_code.append(df_lst.iloc[i]['code'])
            temp_label.append(df_lst.iloc[i]['label'])
            temp_onset.append(df_lst.iloc[i]['onset'])
            temp_cond.append(df_lst.iloc[i]['cond'])
            temp_indx.append(df_lst.iloc[i]['indx'])
    df_tlst['code']=temp_code
    df_tlst['label']=temp_label
    df_tlst['onset']=temp_onset
    df_tlst['cond']=temp_cond
    df_tlst['indx']=temp_indx
    assert len(df_tlst) == expected_tlst, "ERROR: len(df_tlst) != expected tlst"
    # assert len(df_tlst) == len(events_tlst), "ERROR: len(df_tlst) != len(events_tlst)"

    # create dataframe particular for slst
    df_slst = pd.DataFrame(columns=col_names)
    temp_code = []
    temp_label = []
    temp_onset = []
    temp_cond = []
    temp_indx = []
    for i in range(0, len(df_lst)):
        if df_lst.iloc[i]['code'] == 'slst':
            temp_code.append(df_lst.iloc[i]['code'])
            temp_label.append(df_lst.iloc[i]['label'])
            temp_onset.append(df_lst.iloc[i]['onset'])
            temp_cond.append(df_lst.iloc[i]['cond'])
            temp_indx.append(df_lst.iloc[i]['indx'])
    df_slst['code']=temp_code
    df_slst['label']=temp_label
    df_slst['onset']=temp_onset
    df_slst['cond']=temp_cond
    df_slst['indx']=temp_indx
    assert len(df_slst) == expected_slst, "ERROR: len(df_slst) != expected slst"

    print('dataframes created for subject ' + temp)
    print('trials found: ' + str(len(df_tlst)))
    print('sentences found: ' + str(len(df_slst)))
    return nsdata, df_lst, df_plst, df_tlst, df_slst


def create_df_onset(df_tlst):
    """
    Extract just the onsets of given dataframe

    Parameters
    ----------
    df_tlst: must be dataframe extracted from ns event log

    Returns
    -------
    df_tlstS: a new pandas dataframe that only contains onsets
    """
    col_names = ['code','label','onset','cond','indx']
    df_tlstS = pd.DataFrame(columns=col_names)
    temp_code = []
    temp_label = []
    temp_onset = []
    temp_cond = []
    temp_indx = []
    for i in range(0, len(df_tlst)):
        try:
            if df_tlst.iloc[i]['label'] == 'lstS':
                temp_code.append(df_tlst.iloc[i]['code'])
                temp_label.append(df_tlst.iloc[i]['label'])
                temp_onset.append(df_tlst.iloc[i]['onset'])
                temp_cond.append(df_tlst.iloc[i]['cond'])
                temp_indx.append(df_tlst.iloc[i]['indx'])
        except IndexError:
            pass
    df_tlstS['code']=temp_code
    df_tlstS['label']=temp_label
    df_tlstS['onset']=temp_onset
    df_tlstS['cond']=temp_cond
    df_tlstS['indx']=temp_indx
    return df_tlstS


# Things to add: custom labels for imp check, imp index as params
def find_impedances(nsdata):
    """
    Finds impedance onsets and durations based on netstation event log
    ***run 'create_df' before running this

    Parameters
    ----------
    nsdata: list of event logs created from 'create_df'

    Returns
    -------
    imp_onset: list of impedance onset timing in seconds
    imp_offset: list of impedance offset timing in seconds
    imp_dur: list of impedance durations in seconds
    """
    print('finding impedance periods...')
    # first check which version of exp (ts vs st)
    if 'st' in nsdata[0][0][-3:]:
        type = 'st'
    if 'ts' in nsdata[0][0][-3:]:
        type = 'ts'

    # find impedance onset from ns event log
    imp_df = []
    for i in range(0, len(nsdata)):
        if nsdata[i][0] == 'cal+':
            imp_df.append(nsdata[i])
    imp_onset =[]
    for i in range(0, len(imp_df)):
        imp_onset.append(imp_df[i][4])
    imp_onset

    # find impedance offset from ns event log
    if type == 'st':
        imp_offset = []
        indx_val = ['100','200','300']
        for i in range(0, len(nsdata)):
            if nsdata[i][0] == 'prac' and nsdata[i][1] == 'jitr' and nsdata[i][9] == '0':
                imp_offset.append(nsdata[i][4])
            if nsdata[i][0] == 'tral' and nsdata[i][1] == 'jitr' and nsdata[i][9] in indx_val:
                imp_offset.append(nsdata[i][4])
        imp_offset
    if type == 'ts':
        imp_offset = []
        indx_val = ['100','200','300']
        for i in range(0, len(nsdata)):
            if nsdata[i][0] == 'sntn' and nsdata[i][1] == 'jitr' and nsdata[i][9] == '0':
                imp_offset.append(nsdata[i][4])
            if nsdata[i][0] == 'tral' and nsdata[i][1] == 'jitr' and nsdata[i][9] in indx_val:
                imp_offset.append(nsdata[i][4])
        imp_offset

    # convert onset, offset to seconds
    for i in range(0, len(imp_onset)):
        imp_onset[i] = imp_onset[i] * 0.001
    for i in range(0, len(imp_offset)):
        imp_offset[i] = imp_offset[i] * 0.001

    # get impedance duration ie. offset - onset
    imp_dur = []
    for i in range(0, len(imp_onset)):
        imp_dur.append(imp_offset[i] - imp_onset[i])
    imp_dur
    assert len(imp_onset) == len(imp_offset), "ERROR: len of imp_onset is " + str(len(imp_onset) + " while len of imp_offset is " + str(len(imp_offset)) + ". Check for pauses in sessions.")

    print('found ' + str(len(imp_onset)) + ' impedance periods!')
    return imp_onset, imp_offset, imp_dur


def assign_event_id(df, events):
    """
    Update event ids to stim onset/offset by 1 and 2 respectively
    also compares sample number as sanity check (will assert if different)

    Parameters
    ----------
    df: pandas dataframe created from 'create_df'
    events: events arrary from mne.find_events

    Returns
    -------
    events: overwrites on the given events array to update the third column
    """
    print('updating mne event array and double checking sampling onset time...')
    for i in range(0, len(df)):
        # if the sampling time is same, couple them
        if abs(df.iloc[i]['onset'] * 0.2 - events[i][0]) < 1:
        # check whether it's onset or offset, based on last letter of label
            if df.iloc[i]['label'][-1:] == 'S':
                events[i][2] = 1
            if df.iloc[i]['label'][-1:] == 'E':
                events[i][2] = 2
        assert abs(df.iloc[i]['onset'] * 0.2 - events[i][0]) < 1, "ERROR: df sample number different at " + str(i)
    return events


def find_onsets(events):
    """
    Filters and finds just the onset start of event tag from events

    Parameters
    ----------
    events: events arrary from mne.find_events

    Returns
    -------
    events: overwrites on the given events array to update the third column
    """
    print('updating mne event array and double checking sampling onset time...')
    events_onset = np.array([])
    for i in range(0, len(events)):
        if events[i][2] == 1:
            events_onset.append(events[i])
    return events_onset
