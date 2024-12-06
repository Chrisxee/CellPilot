# Data Processing

This folder contains the following files:
- `dataset.py`: The Pytorch dataset used for training our model
- `data_fetching.py`: `len`and `get` methods for each dataset
- `data_utils.py`: DataProcessing and PromptProcessing methods

## Data Fetching
`data_fetching.py` implements `len`and `get` methods for all used datasets. For this we assume our own file structure. More details in the section [Our File Structure](#our-file-structure).
However you can easily implement `len` and `get` for your own dataset. For this have a look at the section [Using your own Cluster/Dataset](#using-your-own-clusterdataset).

### Our File Structure
All of our data is stored in one `data_directory`. Each dataset has a subdirectory in this data_directory. We predefined the file structure for the helmholtz cluster. It is avilable with the keyword `helmholtz` in the cluster variable.


### Using your own Cluster/Dataset
For existing datasets you can change the `len` and `get` methods in DATASET_DICT.
For your own dataset you can implement your own `len` and `get` methods and add them to DATASET_DICT.
`len` should return the number of samples in the dataset.
`get` should return a tuple `(image_path, mask_path)` where `image_path` is the path to the image file and `mask_path` is the path to the mask file.
You should also change the `cluster` variable to `custom`.

## Dataset & Data Utils
You probably don't need to change anything in `dataset.py` and `data_utils.py`.