# Parkinson's Disease Detection with Scalp Electroencephalography (EEG)

## Dataset's Used
[UC San Diego Dataset](https://openneuro.org/datasets/ds002778/versions/1.0.1)

## Paper References and Links
[Reference](https://www.eneuro.org/content/6/3/ENEURO.0151-19.2019)

## Software Requirements
```
python >= 3.7
```

## Repository Structure
```
./
├── LICENSE
├── README.md
└── __main__.py
```

## Software Help
run command `python __main__.py -h`
```
usage: __main__.py [-h] [-c {PD,NONPD,ALL}]

Split EEG data preprocess and create spectrograms

optional arguments:
  -h, --help            show this help message and exit
  -c {PD,NONPD,ALL}, --class {PD,NONPD,ALL}
                        Flag used to determine what class type we want to cretae spectrogram images for
```

# Methods
## Method 1
Attempting to generate spectrogram images for Parkinson's Disease (PD) patients and
NON PD patients

#### Goal
* To predict/classify EEG Readings that contain biomarker indicators of Parkinson's Disease
as appose to EEG reading that do not contain signs of Parkinson's Disease
