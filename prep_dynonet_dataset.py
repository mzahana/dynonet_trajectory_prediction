"""
This script takes csv file of the drone trajectory dataset with the following header
 timestamp,tx,ty,tz,qx,qy,qz,qw
* it takes only the tx,ty,tz points and and populate two numpy arrays,
     u_in: (N, 3*window_size_u)
     y_meas; (N, 3*window_size_y)
* then it saves the prepared dataset to be trained by the DynoNet network
* It is assumed that the data points are sampled evenly in time

* Load u_in and y_meas from the saved file
loaded_data = np.load("data_processed.npz")
u_in_loaded = loaded_data["u_in"]
y_meas_loaded = loaded_data["y_meas"]

print("Loaded u_in shape:", u_in_loaded.shape)
print("Loaded y_meas shape:", y_meas_loaded.shape)

"""
import numpy as np
import csv
import os
import sys
import argparse

def read_csv_files_to_numpy(directory_path):
    csv_files = [file for file in os.listdir(directory_path) if (file.endswith('.csv') or file.endswith('.txt'))]
    if not csv_files:
        print("No CSV files (.cvs , .txt) found in the specified directory.")
        return None

    pos_arrays = {}
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        try:
            # Read CSV using csv module
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                headers = next(csv_reader)  # Read the header row
                pos_arrays[file[:-4]] = np.array([row[0:4] for row in csv_reader], dtype=float)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"Number of datasets: {len(pos_arrays)}")

    return pos_arrays


def process_data(data, dt=0.1, inp_len=10, out_len=5):
    """
    
    Param
    --
    @data is a dictionarry of numpy arrays of the stamped position data
    dt: sampling time in seconds. Default 0.1 second
    inp_len: Length of each input trajectory training sample
    out_len: Length of each output/predicted trajectory

    """
    merged_pos_in=[]
    merged_pos_out = []
    merged_vel_in = []
    merged_vel_out = []
    merged_pos_dataset = []

    for file_name, array_data in data.items():
        # print(f"\nProcessing data from '{file_name}.csv' ...")
        # print(f"Size of dataset {file_name}: {len(array_data)} \n")

        merged_pos_dataset.append(array_data)

        # Estimate velocity
        Pnow = array_data[1:,:]
        Plast = array_data[:-1, :]
        deltas = Pnow-Plast # [dt, dx, dy, dz]
        # print("deltas:", deltas)
        V=np.array([Pnow[:,0], deltas[:,1]/deltas[:,0], deltas[:,2]/deltas[:,0], deltas[:,3]/deltas[:,0] ])
        V = V.T # [vx, vy, vz]

        # Process the dataset into u_in and y_meas arrays
        freq = 1/dt # Sampling freq Hz
        window_size_u = int(inp_len)  # Window size for u_in
        window_size_y = int(out_len)   # Window size for y_meas

        pos_in = []
        vel_in = []
        pos_out = []
        vel_out = []

        if (len(array_data) < (window_size_u+window_size_y)):
            print("Dataset does not have enough points {} < {}".format(len(array_data),window_size_y+window_size_u))
            exit(1)
        # position trajectory
        for i in range(window_size_u, len(array_data) - window_size_y):
            pos_in.append(array_data[i - window_size_u : i, 1:].flatten())
            pos_out.append(array_data[i : i + window_size_y, 1:].flatten())

        pos_in = np.array(pos_in)
        pos_in = pos_in.astype(np.float32)
        pos_out = np.array(pos_out)
        pos_out = pos_out.astype(np.float32)

        merged_pos_in.append(pos_in)
        merged_pos_out.append(pos_out)

        # print(f"position datasets shape in {file_name}: {array_data.shape}")
        # print(f"pos_in shape in {file_name}: {pos_in.shape}")
        # print(f"pos_out shape in {file_name}: {pos_out.shape}")
        # print(f"Type of pos_in in {file_name}: {pos_in.dtype}")
        # print(f"Type of pos_out in {file_name}: {pos_out.dtype}")

        if (len(V) < (window_size_u+window_size_y)):
            print("Velocity dataset does not have enough points {} < {}".format(len(V),window_size_y+window_size_u))
            exit(1)
        # velocity trrajectory
        for i in range(window_size_u, len(V) - window_size_y):
            vel_in.append(V[i - window_size_u : i, 1:].flatten())
            vel_out.append(V[i : i + window_size_y, 1:].flatten())

        vel_in = np.array(vel_in)
        vel_in = vel_in.astype(np.float32)
        vel_out = np.array(vel_out)
        vel_out = vel_out.astype(np.float32)

        merged_vel_in.append(vel_in)
        merged_vel_out.append(vel_out)

        # print(f"velocity datasets shape in {file_name}: {V.shape}")
        # print(f"vel_in shape in {file_name}: {vel_in.shape}")
        # print(f"vel_out shape in {file_name}: {vel_out.shape}")
        # print(f"Type of vel_in in {file_name}: {vel_in.dtype}")
        # print(f"Type of vel_out in {file_name}: {vel_out.dtype}")


    # merge all lists into numpy arrays
    merged_pos_dataset_np = np.concatenate(merged_pos_dataset, axis=0)
    merged_pos_in_np = np.concatenate(merged_pos_in, axis=0)
    merged_pos_out_np = np.concatenate(merged_pos_out, axis=0)
    merged_vel_in_np = np.concatenate(merged_vel_in, axis=0)
    merged_vel_out_np = np.concatenate(merged_vel_out, axis=0)

    print(f"Number of samples in the concatenated original position dataset = {len(merged_pos_dataset_np)}")
    print(f"Number of samples of processed datasets = {len(merged_pos_in_np)}")
    print(f"shape of merged_pos_in_np: {merged_pos_in_np.shape}")
    print(f"Sampling time: {dt} seconds")
    print(f"Input trajectory lenght: {inp_len} points")
    print(f"Output trajectory length: {out_len} points")

    return merged_pos_dataset_np, merged_pos_in_np, merged_pos_out_np, merged_vel_in_np, merged_vel_out_np

# # Load the CSV file into a NumPy array
# filename = "indoor_forward_7_davis_with_gt.txt"
# file_path = os.path.join("resampled_100ms_dataset", filename)
# out_file=""
# dt = np.array(0.1)

# with open(file_path, "r") as file:
#     csv_reader = csv.reader(file)
#     headers = next(csv_reader)  # Read the header row
#     pos_dataset = np.array([row[0:4] for row in csv_reader], dtype=float)

# # Estimate velocity
# Pnow = pos_dataset[1:,:]
# Plast = pos_dataset[:-1, :]
# deltas = Pnow-Plast # [dt, dx, dy, dz]
# # print("deltas:", deltas)
# V=np.array([Pnow[:,0], deltas[:,1]/deltas[:,0], deltas[:,2]/deltas[:,0], deltas[:,3]/deltas[:,0] ])
# V = V.T # [vx, vy, vz]


# # Process the dataset into u_in and y_meas arrays
# freq = 10 # Sampling freq Hz
# window_size_u = freq  # Window size for u_in
# window_size_y = int(freq/2)   # Window size for y_meas

# pos_in = []
# vel_in = []
# pos_out = []
# vel_out = []

# if (len(pos_dataset) < (window_size_u+window_size_y)):
#     print("Dataset does not have enough points {} < {}".format(len(dataset),window_size_y+window_size_u))
#     exit(1)
# # position trajectory
# for i in range(window_size_u, len(pos_dataset) - window_size_y):
#     pos_in.append(pos_dataset[i - window_size_u : i, 1:].flatten())
#     pos_out.append(pos_dataset[i : i + window_size_y, 1:].flatten())

# pos_in = np.array(pos_in)
# pos_in = pos_in.astype(np.float32)
# pos_out = np.array(pos_out)
# pos_out = pos_out.astype(np.float32)

# print("position datasets shape:", pos_dataset.shape)
# print("pos_in shape:", pos_in.shape)
# print("pos_out shape:", pos_out.shape)
# print("Type of pos_in:", pos_in.dtype)
# print("Type of pos_out", pos_out.dtype)

# if (len(V) < (window_size_u+window_size_y)):
#     print("Velocity dataset does not have enough points {} < {}".format(len(V),window_size_y+window_size_u))
#     exit(1)
# # velocity trrajectory
# for i in range(window_size_u, len(V) - window_size_y):
#     vel_in.append(V[i - window_size_u : i, 1:].flatten())
#     vel_out.append(V[i : i + window_size_y, 1:].flatten())

# vel_in = np.array(vel_in)
# vel_in = vel_in.astype(np.float32)
# vel_out = np.array(vel_out)
# vel_out = vel_out.astype(np.float32)

# print("velocity datasets shape:", V.shape)
# print("vel_in shape:", vel_in.shape)
# print("vel_out shape:", vel_out.shape)
# print("Type of vel_in:", vel_in.dtype)
# print("Type of vel_out", vel_out.dtype)



# # Save to a file
# output_dir = "dynonet_datasets"

# # Specify the directory name you want to check/create
# directory_name = "my_directory"

# # Get the current directory path
# current_directory = os.getcwd()

# # Concatenate the current directory path with the directory name
# directory_path = os.path.join(current_directory, output_dir)

# # Check if the directory already exists
# if not os.path.exists(directory_path):
#     # Create the directory if it doesn't exist
#     os.makedirs(directory_path)
#     print("Directory created:", directory_path)
# else:
#     print("Directory already exists:", directory_path)


# output_file = directory_path+"/indoor_forward_7_davis.npz"
# np.savez(output_file, pos_dataset=pos_dataset, pos_in=pos_in, pos_out=pos_out, vel_in=vel_in, vel_out=vel_out, dt=dt)
# print("Arrays saved to", output_file)

# # Load data example
# # loaded_data = np.load("data_processed.npz")
# # u_in_loaded = loaded_data["u_in"]
# # y_meas_loaded = loaded_data["y_meas"]

# # print("Loaded u_in shape:", u_in_loaded.shape)
# # print("Loaded y_meas shape:", y_meas_loaded.shape)


def main():
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read datasets from CSV files from a directory and prepare them fro Dynonet.")
    parser.add_argument("input_directory", help="Path to the directory containing CSV files.")
    parser.add_argument("-o", "--output_directory", help="Optional path to save the concatenated numpy array as a CSV file.")
    parser.add_argument("-of", "--output_file", help="Optional name of the file that contains the processed dataset as .npz .")
    parser.add_argument("-iL", "--input_length", type=int, help="Lenght of each input trajectory sample")
    parser.add_argument("-oL", "--output_length", type=int, help="Lenght of each output/predicted trajectory")
    parser.add_argument("-dt", "--sampling_time", type=float, default=0.1, help="Sampling time in seconds")
    args = parser.parse_args()

    if args.sampling_time:
        dt = args.sampling_time
    if args.input_length:
        input_length = args.input_length
    else:
        input_length = 10
    if args.output_length:
        output_length = args.output_length
    else:
        output_length = 5

    directory_path = args.input_directory
    data = read_csv_files_to_numpy(directory_path)

    merged_pos_dataset_np, merged_pos_in_np, merged_pos_out_np, merged_vel_in_np, merged_vel_out_np = process_data(data=data, dt=0.1, inp_len=input_length, out_len=output_length)

    # Save the concatenated numpy array to a CSV file if output directory is provided
    if args.output_directory:
        output_file = args.output_directory+"/dynonet_dataset.npz"

        if args.output_file:
            output_file = args.output_directory+"/"+args.output_file+".npz"

        np.savez(output_file, pos_dataset=merged_pos_dataset_np, pos_in=merged_pos_in_np, pos_out=merged_pos_out_np, vel_in=merged_vel_in_np, vel_out=merged_vel_out_np, dt=dt)
        print("Arrays saved to", output_file)
