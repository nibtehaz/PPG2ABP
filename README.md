# PPG2ABP 
#### Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks

This repository contains the original implementation of "PPG2ABP : Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks" in Keras (Tensorflow as backend).


## Overview

Cardiovascular diseases are one of the most severe causes of mortality, annually taking a heavy toll on lives worldwide. Continuous monitoring of blood pressure seems to be the most viable option, but this demands an invasive process, introducing several layers of complexities and reliability concerns due to non-invasive techniques not being accurate. This motivates us to develop a method to estimate the continuous arterial blood pressure (ABP) waveform through a non-invasive approach using Photoplethysmogram (PPG) signals. We explore the advantage of deep learning, as it would free us from sticking to ideally shaped PPG signals only by making handcrafted feature computation irrelevant, which is a shortcoming of the existing approaches. Thus, we present PPG2ABP, a two-stage cascaded deep learning-based method that manages to estimate the continuous ABP waveform from the input PPG signal with a mean absolute error of 4.604 mmHg, preserving the shape, magnitude, and phase in unison. However, the more astounding success of PPG2ABP turns out to be that the computed values of Diastolic Blood Pressure (DBP), Mean Arterial Pressure (MAP), and Systolic Blood Pressure (SBP) from the estimated ABP waveform outperform the existing works under several metrics (mean absolute error of 3.449 ± 6.147 mmHg, 2.310 ± 4.437 mmHg, and 5.727 ± 9.162 mmHg, respectively), despite that PPG2ABP is not explicitly trained to do so. Notably, both for DBP and MAP, we achieve Grade A in the BHS (British Hypertension Society) Standard and satisfy the AAMI (Association for the Advancement of Medical Instrumentation) standard.

## Codes

The codes for the PPG2ABP pipeline can be found in the Codes directory.

## Demo

A demo can be found in [here](https://github.com/nibtehaz/PPG2ABP/blob/master/codes/PPG2ABP.ipynb)


## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

[MIT license](https://github.com/nibtehaz/MultiResUNet/blob/master/LICENSE)


## Citation Request

If you use ***PPG2ABP*** in your project, please cite the following paper

```
@article{ibtehaz2020ppg2abp,
  title={PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks},
  author={Ibtehaz, Nabil and Rahman, M Sohel},
  journal={arXiv preprint arXiv:2005.01669},
  year={2020}
}
```
