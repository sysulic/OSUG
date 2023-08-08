# OSUG 

---
**LTL Satisfiability Checking via Graph Representation Learning**

## Requirements

---

For running the code:
+ ltlf2dfa
+ matplotlib
+ numpy
+ torch_geometric

Try ```pip install torch_geometric```
If fail follow the following steps:

+ step in the website : [pyg wheel](https://data.pyg.org/whl/)
    find the correct version of your torch
    for example: ```torch-1.12.1+cu116```

+ step in and download all wheels we need:
    if using python=3.9, sys=linux, then
    ``` 
        pyg_lib-0.1.0+pt112cu116-cp39-cp39-linux_x86_64.whl
        torch_cluster-1.6.0+pt112cu116-cp39-cp39-linux_x86_64.whl
        torch_scatter-2.1.0+pt112cu116-cp39-cp39-linux_x86_64.whl
        torch_sparse-0.6.16+pt112cu116-cp39-cp39-linux_x86_64.whl
        torch_spline_conv-1.2.1+pt112cu116-cp39-cp39-linux_x86_64.whl 
    ```
+ Install all the wheels downloaded and run the command ```pip install torch_geometric```.

## Dataset Prepare

---
+ Unzip '*.zip' files in dir 'data'.
+ The first time preprocessing training data will cost a lot of time and if you want to only get the preprocessed data, you can modify the dir in data_sc.py/data_sv.py and run the python file.

## Satisfiability Checking
---

### How to train and evaluate

``` python train_sc.py ```

### How to test

+ Find the best model path: test record dir and saved best model name
+ Choose the test dataset: test data dir and test dataset
+ Run ``` python test_sc.py --trp <test record dir> --sbm <saved best model name> --dt <test data dir> --td <test dataset> ```

## Satisfiable Trace Predicting
---

### How to train

``` python train_sv.py --is_train 1```

### How to test

+ Find the best model path: test record dir and saved best model name
+ Choose the test dataset: test data dir and test dataset
+ Run ``` python train_sv.py --is_train 0 --trp <test record dir> --sbm <saved best model name> --dt <test data dir> --td <test dataset> ```
