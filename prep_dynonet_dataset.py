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

# Load the CSV file into a NumPy array
filename = "indoor_forward_3_davis_with_gt.txt"
file_path = os.path.join("resampled_100ms_dataset", filename)
out_file=""
dt = np.array(0.1)

with open(file_path, "r") as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)  # Read the header row
    pos_dataset = np.array([row[0:4] for row in csv_reader], dtype=float)

# Estimate velocity
Pnow = pos_dataset[1:,:]
Plast = pos_dataset[:-1, :]
deltas = Pnow-Plast # [dt, dx, dy, dz]
# print("deltas:", deltas)
V=np.array([Pnow[:,0], deltas[:,1]/deltas[:,0], deltas[:,2]/deltas[:,0], deltas[:,3]/deltas[:,0] ])
V = V.T # [vx, vy, vz]


# Process the dataset into u_in and y_meas arrays
freq = 10 # Sampling freq Hz
window_size_u = freq  # Window size for u_in
window_size_y = int(freq/2)   # Window size for y_meas

pos_in = []
vel_in = []
pos_out = []
vel_out = []

if (len(pos_dataset) < (window_size_u+window_size_y)):
    print("Dataset does not have enough points {} < {}".format(len(dataset),window_size_y+window_size_u))
    exit(1)
# position trajectory
for i in range(window_size_u, len(pos_dataset) - window_size_y):
    pos_in.append(pos_dataset[i - window_size_u : i, 1:].flatten())
    pos_out.append(pos_dataset[i : i + window_size_y, 1:].flatten())

pos_in = np.array(pos_in)
pos_in = pos_in.astype(np.float32)
pos_out = np.array(pos_out)
pos_out = pos_out.astype(np.float32)

print("position datasets shape:", pos_dataset.shape)
print("pos_in shape:", pos_in.shape)
print("pos_out shape:", pos_out.shape)
print("Type of pos_in:", pos_in.dtype)
print("Type of pos_out", pos_out.dtype)

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

print("velocity datasets shape:", V.shape)
print("vel_in shape:", vel_in.shape)
print("vel_out shape:", vel_out.shape)
print("Type of vel_in:", vel_in.dtype)
print("Type of vel_out", vel_out.dtype)



# Save to a file
output_dir = "dynonet_datasets"

# Specify the directory name you want to check/create
directory_name = "my_directory"

# Get the current directory path
current_directory = os.getcwd()

# Concatenate the current directory path with the directory name
directory_path = os.path.join(current_directory, output_dir)

# Check if the directory already exists
if not os.path.exists(directory_path):
    # Create the directory if it doesn't exist
    os.makedirs(directory_path)
    print("Directory created:", directory_path)
else:
    print("Directory already exists:", directory_path)


output_file = directory_path+"/indoor_forward_3_davis.npz"
np.savez(output_file, pos_dataset=pos_dataset, pos_in=pos_in, pos_out=pos_out, vel_in=vel_in, vel_out=vel_out, dt=dt)
print("Arrays saved to", output_file)

# Load data example
# loaded_data = np.load("data_processed.npz")
# u_in_loaded = loaded_data["u_in"]
# y_meas_loaded = loaded_data["y_meas"]

# print("Loaded u_in shape:", u_in_loaded.shape)
# print("Loaded y_meas shape:", y_meas_loaded.shape)