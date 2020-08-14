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
├── __main__.py
├── datautils
│   ├── data_loader.py
│   ├── stft.py
│   └── wavelet_transform.py
└── models
    ├── ensemble.py
    ├── stft_model.py
    └── wavelet_model.py

2 directories, 9 files
```

# Software
run command `python __main__.py -h`

Main Script is used to generate spectrogram images, or scalogram images depending on parameters passed to script.
It uses the directory **./data** as to root directory for the EEG readings. Directory is not pushed to repository,
but data can be downloaded from the link above on **UC San Diego Dataset**
```
usage: __main__.py [-h] -c {PD_OFF,PD_ON,NONPD,ALL} [-s] [-w] -o OUTPUT_DIR

Split EEG data preprocess and create spectrograms

optional arguments:
  -h, --help            show this help message and exit
  -c {PD_OFF,PD_ON,NONPD,ALL}, --class {PD_OFF,PD_ON,NONPD,ALL}
                        Flag used to determine what class type we want to cretae spectrogram images for
  -s, --stft            Flag used to utilize the short-time fourier transform in data processing
  -w, --wave            Flag used to utilize wavelet transform in data processing
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Flag used to determine the root output path to place images
```

## Models
#### models/wavelet_model.py
The **wavelet_model** script is used to generate a model for a single channel of the EEG recordings.
This model utilizes the scalogram images developed via the wavelet transform.


#### models/stft_model.py
The **stft_model** script is used to generate a model for a single channel of the EEG recordings.
The model utilizes the spectrogram images developed via short-time fourier transform.


#### models/ensemble.py
The **ensemble** script is used to generate 1 model for each channel/leads of the EEG recording headset.
For example with a 32 channel headset will result in 32 different models developed. This is in hopes to
develop an ensemble model that would utilize the models of each channel, to make it's final classification/prediction.
help on running the ensemble, by running command `python models/ensemble.py -h`


```
usage: ensemble.py [-h] -i IMAGE_SIZE -s SIZE -e EPOCHS -o OUTPUT_DIR -d DATA_ROOT

Train a model to classify spectrograms

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_SIZE, --image-size IMAGE_SIZE
  -s SIZE, --set SIZE   Flag used to determine the amount of experiments to import for train/test data
  -e EPOCHS, --epochs EPOCHS
                        Flag used to determine the number of epochs for training
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory where to output all models created for each channel
  -d DATA_ROOT, --data-root DATA_ROOT
                        root of directory for training testing data images
```


#### Goal
* To predict/classify EEG Readings that contain biomarker indicators of Parkinson's Disease
as appose to EEG reading that do not contain signs of Parkinson's Disease. There are currently
3 different classes we are looking at:

```
- NON Parkinson's Disease Patients
- Parkinson's Disease Patients OFF medication
- Parkinson's Disease Patients ON medication
```

## Resources
```
https://bigdatawg.nist.gov/HackathonTutorial.html

https://raphaelvallat.com/bandpower.html
```
