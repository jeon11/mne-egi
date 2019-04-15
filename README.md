## Introduction
The walkthrough.py code suggests for general pipeline of EEG preprocessing to ERP plotting using the MNE toolbox (https://mne-tools.github.io/stable/index.html).

The walkthrough_basics.ipynb runs through the basics from reading in raw instace and creating metadata using custom codes for the experiment to creating epochs and plotting evoked responses by condition using the created metadata.

The walkthrough_advanced.ipynb runs independent component analysis to reject bad ICs and uses automated autoreject module to further clean the data. With the cleaned epochs and evoked response, we compare the results to the original data processed from walkthrough_basics.

To run the code locally, see dependencies. 


### dependencies:
1. raw data and events export text: Link [download from Google Drive ~500MB](https://drive.google.com/file/d/1W2UFu_6H4HzFF2DALAxfmr0BNSj7pEok/view?usp=sharing)
  - Download the raw file and move to /data/ directory


2. MNE toolbox: https://mne-tools.github.io/stable/getting_started.html

3. autoreject: http://autoreject.github.io/#installation

4. pandas: https://pandas.pydata.org

5. matplotlib: https://matplotlib.org/users/installing.html

6. Scipy: https://www.scipy.org
