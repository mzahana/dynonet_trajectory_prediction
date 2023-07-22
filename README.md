# dynonet_trajectory_prediction
Scripts to prepare datasets, and train/test a DynoNet network for trajectory prediction.

# Steps
* Get original dataset as a CSV file. It should have the following strucure.
    ```csv
    timestamp, tx,ty,tz
    ```
    You can find datasets in [here](https://fpv.ifi.uzh.ch/datasets/)
* It is requried that the data is equally spaced in time (with constant sampling frequency). If it is not the case, you can use`python3 resample_data.py` to resample data at equidistant times. Modify the script to read the correct dataset and set the required `dt`. The input dataset are expected to be inside the `original_datasets` directory

* Run the `prep_dynonet_dataset.py` script to process the original datasets to one that is suitable for training a Dynonet network. It takes a directory of multiple trajectories and will concatenate them all into one dataset after preparing the input/output trajectories from the indvidual ones. Example usage:
    ```bash
    python3 prep_dynonet_dataset.py resampled_100ms_dataset/datasets -o resampled_100ms_dataset -iL 10 -oL 5 -dt 0.1
    ```
    * First argument is the path to the directory containing the datasets with extensions `.txt` or `.csv` ONLY
    * `-o` is the output directory of the concatenated dataset
    * `-iL` is the input trajectory length (int)
    * `-oL` Output trajectory length (int)
    * `-dt` sampling time that is used in the datasets

* (optional) You can use `merge_datasets.py` to merge multiple datasets of different trajectories. Note that `prep_dynonet_dataset.py` already merges all datasets in the specified directory. However, the `merge_datasets.py` can be used to merge specific datasets.

* Use the `train.py` file to train the dynonet network as follows.
    ```bash
    python3 train.py resampled_100ms_dataset/dynonet_dataset.npz -it 10000 
    ```
    * First argument `resampled_100ms_dataset/dynonet_dataset.npz` is the path to the dataset file
    * `-it` maximum number of training iterations
    * model is saved in the same directory of the `train.py` file inside the `models` directory

* To test the trained model on a dataset `*.npz`
    ```bash
    python3 test.py dynonet_datasets/indoor_45_4_davis.npz -m models/drone_trajecrory_model -s 100
    ```
    * First argument is the path to the dataset `*.npz` file
    * `-m` is the directory that has the models (`G1`, and `F1`) 
    * `-s` is the index of the sample trajectory to test in `*.npz` dataset.