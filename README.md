This repository contains most of the (anonymised) files used for the pre-processing and analysis of data for my thesis (available at doi.org/10.5281/zenodo.23883955).

**Chapter One**
*Krajjat*
Python module for the pre-processing, visualisation and analysis of motion capture data in relation to speech (Full C1). In its own repository:
GitHub repository: https://github.com/RomainPastureau/Krajjat
Documentation: https://krajjat.readthedocs.io/en/latest/index.html
PyPI page: https://pypi.org/project/krajjat/

**Chapter Two**
*cartoon_randomiser_kinect*	
Short scripts to list and randomise the cartoons used in the first production study (C2-2.1.2).

*countup*
Pygame script displaying a countdown and a timer as a visual aid during the first production study (C2-2.1.4).

*kinect_preprocessing*
Contains all the scripts used to pre-process the Kinect data in the first production study (C2-2.1.5): correction of a single specific file (A01), assembling of individual frames (A02), pre-processing of mocap sequences (A03), pre-processing of audio recordings (A04), and dataframe generation (A05).

*kinect_analysis*
Scripts used for the computationally intensive analyses run on a cluster (B scripts: cross-correlation, coherence), statistical analyses and figure generation (C and D scripts). See C2-2.1.6.

*cartoon_randomiser_qualisys*
Short scripts to randomise the cartoons used in the second production study (C2-2.2.2), and to find the exact time codes of the cartoon excerpts from the original cartoon videos.

*qualisys_preprocessing*
Contains all the scripts used to pre-process the Qualisys data in the second production study (C2-2.2.5): pre-processing of audio recordings (A01) and mocap sequences (A02), generation of audio derivatives (A03), and dataframe generation (A04).

*qualisys_analysis*
Scripts used for the computationally intensive analyses run on a cluster (B scripts: cross-correlation, coherence), statistical analyses and figure generation (C and D scripts). See C2-2.2.6.

**Chapter Three**
*pretest_comprehension*
jsPsych experiment scripts used in the pre-test of the comprehension experiment (C3-2.1.3).

*audiovisual_preprocessing*
Various bash, MATLAB and Python functions aiming at pre-processing the videos (luminance correction and cropping) and creating the stimuli for the comprehension experiment (adding noise, and creating unimodal and multimodal stimuli; see C3-2.2.2).

*psychtoolbox_experiment*
Psychtoolbox experiment script used in the main test of the comprehension experiment (C3-2.2.4). The directory contains example stimuli.

*meg_preprocessing*
Scripts containing the full pre-processing MEG pipeline, including BIDS creation, bad channel detection, Maxfiltering, ICA-filtering, empty room pre-processing, co-registration with MRI, and delay correction (see C3-2.2.5).

*trf_analysis*
Scripts relating to the TRF predictor creation and model comparisons (see C3-2.2.6).

*pretest_analysis*
R script for the analysis of the pre-test data (C3-3.1).

*meg_behavioural_analysis*
JASP file related to the analysis of the participants’ answers during the MEG experiment (see C3-3.2.1).

**Supplementary materials**
*find_delay*
Python module developed during the PhD for finding the delay between two arrays. It can be used for aligning two audio files, and was used during the PhD to find the delay in MEG experiments between the Psychtoolbox trigger and the audio recorded on the MISC channel. In its own repository:
GitHub repository: https://github.com/RomainPastureau/find_delay
Documentation: https://find-delay.readthedocs.io/en/latest/?badge=latest
PyPI page: https://pypi.org/project/find-delay/
