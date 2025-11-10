# NICE: Neural Implicit Craniofacial Model for Orthognathic Surgery Prediction
This is the official code repository of our work "NICE: Neural Implicit Craniofacial Model for Orthognathic Surgery Prediction" accepted by AAAI 2026.

## Installation
You can create an anaconda environment called `NICE` using
```
conda create -n "NICE" python=3.9
conda activate NICE
```

To install this repository and its requirements in development mode run
```
pip install -e .
```

If you plan to run a model you will need have a GPU-enabled `pytorch` version running. 
For our experiemnts we used `pytorch 1.13` and `CUDA 11.6`.  




## Data structure
### Shape Data
```dataset```  used for the shape module is organized into separate directories for training, validation, and testing. Each folder is numerically labeled (e.g., 000) to represent an individual sample. For each sample, the following files are included:

- `softtissue.obj`,  `maxilla.obj` and `mandible.obj`: These files contain the ground truth (GT) data of the soft tissue, maxilla and mandible, respectively, which are obtained through CT scans.
  
- `sft_x_face.npy`, `mxl_x_face.npy`, and `mdb_x_face.npy`: These files correspond to the front surface points sampled from the soft tissue and the maxilla and mandible, where 'x' represents the sample index.

- `sft_x_non_face.npy` and `mxl_x_non_face.npy`: These files contain the back surface points sampled from the soft tissue and the maxilla, where 'x' represents the sample index.

### Surgery Data
```surgerydataset``` used for the surgery module is organized into separate directories for training, validation, and testing. Each folder is numerically labeled (e.g., 000) to represent an individual sample. For each sample, the following files are included:
- `ori_sft_pre.obj`,  `ori_mxl_pre.obj` and `ori_mdb_pre.obj`: These files contain the preoperative ground truth (GT) data of the soft tissue, maxilla and mandible, respectively, which are obtained through CT scans.
- `ori_sft_post.obj`,  `ori_mxl_post.obj` and `ori_mdb_post.obj`: These files contain the postoperative ground truth (GT) data of the soft tissue, maxilla and mandible, respectively, which are obtained through CT scans.
- `sft_corresp_x.npy`, `mxl_corresp_x.npy`, and `mdb_corresp_x.npy_x_face.npy`: These files contain the preoperative and postoperative coordinates of sampling points on the soft tissue, maxilla, and mandible, respectively, where 'x' represents the sample index. These points share the same topology, allowing for accurate comparison of the anatomical structures before and after the surgical procedure.
- `sft_corresp_normals_x.npy`, `mxl_corresp_normals_x.npy`, and `mdb_corresp_normals_x.npy_x_face.npy`: These files contain the preoperative and postoperative normal vectors of sampling points on the soft tissue, maxilla, and mandible, respectively, where 'x' represents the sample index. These points share the same topology, allowing for accurate comparison of the anatomical structures before and after the surgical procedure.




## Training 

NICE is trained in two stages.
### Stage 1 - Learning Shape Module

First you need to train the shape module using:
``` 
python scripts/train/train_shape.py -cfg_file scripts/config/NICE_shape.yaml -local -exp_name SHAPE_EXP_NAME 
```

### Stage 2 - Learning Surgery Module

Afterwards, the surgery module can be trained by:

```
python scripts/train/train_surgery.py -cfg_file scripts/config/NICE_surgery.yaml -exp_name SURGERY_EXP_NAME -mode compress
```

## Test


### Test Shape Module
You can test the shape module using
``` 
python scripts/fit/fit_shape.py -cfg_file scripts/config/NICE_shape.yaml -local 
```

### Test Surgery Module
You can test the surgery module using
``` 
python scripts/fit/fit_surgery.-cfg_file scripts/config/NICE_surgery.yaml -exp_name SURGERY_FIT_NAME -mode compress
```

## Predict
You can predict the postoperative face using
```
python scripts/fit/predict_face.py -cfg_file scripts/config/NICE_surgery.yaml -exp_name SURGERY_PREDICT_NAME -mode compress
```

## Data Availability
Due to the data confidentiality agreement with the hospital, we can only provide a few demo cases. The complete dataset will be made publicly available in the future.
