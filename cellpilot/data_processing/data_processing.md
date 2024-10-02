# Data Processing

This folder contains the following files:
- `dataset.py`: The Pytorch dataset used for training our model
- `data_fetching.py`: `len`and `get` methods for each dataset
- `data_utils.py`: DataProcessing and PromptProcessing methods

Detailed Description:
## Dataset

## Data Fetching
`data_fetching.py` implements `len`and `get` methods for all used datasets. For this we assume our own file structure. More details in the section [Our File Structure](#our-file-structure).
However you can easily implement `len` and `get` for your own dataset. For this have a look at the section [Using your own Dataset](#using-your-own-dataset).

### Our File Structure
All of our data is stored in one `data_directory`. Each dataset has a subdirectory in this data_directory. The individual datasets are stored as follows:

#### BCSS
```
BCSS
|--- 0_Public-data-Amgad2019_0.25MPP
    |--- masks
    |--- rgbs_colorNormalized
```
#### CellSeg
```

```

### Using your own Dataset
Entry in DATASET_DICT
Implement len and get
## Data Utils