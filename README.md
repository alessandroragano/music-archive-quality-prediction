# Quality Assessment Music Archives (QAMA)

### Purpose
QAMA is a reference-free (non-intrusive) quality metric for digitised music archives. The QAMA model is useful to predict the perceived audio quality of vinyl recordings. 
The preservation of cultural heritage through audio archives is primarily focused on the process of digitization, which involves curating audio collections from various analog media formats like wax cylinders, vinyl discs, and 78 RPMs. The purpose of QAMA is to improve accessibility and usability of audio archives.
For example, it can be used to retrieve the best quality items or to detect low-quality old vinyl recordings in a collection. The model is optimised for real-world recordings. 

### Vinylset
The QAMA model has been trained with [Vinylset](https://github.com/alessandroragano/vinylset) which includes real-world recordings of several genres e.g. classical, jazz, electronic. We collected 620 tracks extracted from original vinyl recordings and labelled them with mean opinion score (MOS). 

### More Information
For additional information on QAMA and Vinylset see the ICASSP'23 paper (link coming soon). 

For more information on quality assessment for digital audio archives see the [JAES paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/76602/Benetos%20Automatic%20Quality%20Assessment%202022%20Accepted.pdf?sequence=2).

## Installation
First, clone the repo to your local machine.
To use QAMA install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) and then create the virtual environment:
```
mkvirtualenv -p python3.9 QAMA
```
This will create a virtual environment QAMA. Then install the requirements:
```
pip install -r requirements.txt
```
Activate the virtual environment:
```
workon QAMA
```
## Usage
QAMA has been evaluated with a 3-fold cross-validation. We provide the 3 models plus a model trained using the full Vinylset.
* ```qama_fold1.pt```
* ```qama_fold2.pt```
* ```qama_fold3.pt```
* ```qama_full.pt```

To predict the quality of your music collection, organise files in a directory and run:

```python predict.py --data_dir /path/to/dir/collection```

The script creates a csv file in ```prediction_files``` with date time format ```DD-MM-YYYY_hh-mm-ss_qama.csv```
The csv files includes predictions of the 4 models and the average of the 3 cross-validation models. This is defined with the column ```Mean CV```.
### Full Model
The model ```qama_full.pt``` is trained using all the Vinylset tracks. Unlike the cross-validation models, the full model has not been evaluated in the ICASSP paper although it might be more accurate. 
Informal tests show aligned results with the cross-validation models.

You can set ```full_model=False``` if you do not want to use QAMA full.

### Correct usage
QAMA has been evaluated for vinyl degradations and using files around 10 seconds. The model accepts variable input length but it was not evaluated for very long tracks. 

## Paper and license
If you use QAMA or Vinylset please cite this paper: 

A. Ragano, E. Benetos, and A. Hines "Audio Quality Assessment of Vinyl Music Collections using Self-Supervised Learning", in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2023, (link coming soon).


The code is licensed under MIT license.

