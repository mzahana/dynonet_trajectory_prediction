"""
Trains a dynoNet on drone trajectories
- Input is a trajectory of position points px,py,pz over multiple discrete times
    - Example: [px0,py0,pz0, px1,py1,pz1, ..., pxN,pyN,pzN]
- Output is a trajectory over a discrete times with the same input frequency
    - Example: [px0,py0,pz0, px1,py1,pz1, ..., pxM,pyM,pzM]

- Each input trajectory of length N points (x,y,z) results in output (predicted) trajectory of length M points(x,y,z)

- Input dataset is assumed to have two numpy arrays named u_in, y_meas
- u_in: (N, 3*Tin). Tin is the number of input time steps
- y_meas: (N, 3*Tout) Tout is the number of output time steps
- N is the number of data points in the dataset
"""

import torch
import pandas as pd
import numpy as np
import os
from dynonet.lti import MimoLinearDynamicalOperator, SisoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity, MimoStaticNonLinearity
import time
import dynonet.metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test Dynonet.")
    parser.add_argument("data_path", help="Path to the data stored as .npz .")
    parser.add_argument("-m", "--model_dir", help="Directory path where the model is saved .")
    parser.add_argument("-s", "--sample_index", type=int, default=0, help="Index of data sample to test")
    args = parser.parse_args()

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]

    # Extract data
    try:
        loaded_data = np.load(args.data_path)
    except Exception as e:
        print("Error in reading dataset: {}".format(e))
        exit(1)

    # original_file="indoor_forward_3_davis_with_gt.txt"
    # try:
    #     original_df = pd.read_csv(original_file,delimiter=",")
    # except Exception as e:
    #     print("Could not read original dataset {} : ".format(original_file, e))
    #     exit(1)

    
    pos_dataset = loaded_data["pos_dataset"]
    dt = loaded_data["dt"]
    pos_in = loaded_data["pos_in"]
    u = loaded_data["vel_in"]
    y = loaded_data["vel_out"]

    # Number of input data points
    u_N = u.shape[1]
    y_N = y.shape[1]

    # # Model blocks
    # G1 = MimoLinearDynamicalOperator(u_N, u_N, n_b=2, n_a=2, n_k=1)
    # # Static sandwitched non-linearity
    # F1 = MimoStaticNonLinearity(u_N, y_N, activation='tanh')
    # G2 = MimoLinearDynamicalOperator(y_N, y_N, n_b=2, n_a=2, n_k=0)

    # F1 = MimoStaticNonLinearity(u_N, y_N, activation='tanh')
    # G1 = MimoLinearDynamicalOperator(y_N, y_N, n_b=2, n_a=3, n_k=0)
    # F2 = MimoStaticNonLinearity(y_N, y_N, activation='tanh')

    factor=5
    F1 = MimoStaticNonLinearity(u_N, y_N*factor, activation='tanh')
    F2 = MimoStaticNonLinearity(y_N*factor, y_N, activation='tanh')
    G1 = MimoLinearDynamicalOperator(y_N, y_N, n_b=2, n_a=3, n_k=0)
    F3 = MimoStaticNonLinearity(y_N, y_N*factor, activation='tanh')
    F4 = MimoStaticNonLinearity(y_N*factor, y_N, activation='tanh')

    # # Load identified model parameters
    model_name = 'drone_trajecrory_model'
    model_folder = os.path.join("models", model_name)
    model_folder = args.model_dir
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    F1.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
    F2.load_state_dict(torch.load(os.path.join(model_folder, "F2.pkl")))
    F3.load_state_dict(torch.load(os.path.join(model_folder, "F3.pkl")))
    F4.load_state_dict(torch.load(os.path.join(model_folder, "F4.pkl")))
    # G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))

    # # Model structure
    def model(u_in):
        # y_lin_1 = G1(u_in)
        # y_nl = F1(y_lin_1)
        # y_pred = G2(y_nl)

        y_nl_1 = F1(u_in)
        y_nl_2 = F2(y_nl_1)
        y_lin = G1(y_nl_2)
        y_nl_3 = F3(y_lin)
        y_nl_4 = F4(y_nl_3)
        # y_hat = torch.cumsum(v_hat, dim=1) * ts
        return y_nl_4
    
    # # In[Simulate]
    u_fit_torch = torch.tensor(u[None, :, :])
    v_hat = model(u_fit_torch)

    # # In[Detach]
    v_hat = v_hat.detach().numpy()[0, :, :]
    print("v_hat shape: ", v_hat.shape)
    print(f"Ground truth vel_out shape: {y.shape}")

    # Reshape u_in and y_meas to separate tx, ty, tz
    pos_in_tx = pos_in[:, 0::3]
    pos_in_ty = pos_in[:, 1::3]
    pos_in_tz = pos_in[:, 2::3]
        
    v_hat_x = v_hat[:, 0::3]
    v_hat_y = v_hat[:, 1::3]
    v_hat_z = v_hat[:, 2::3]

    # # ground truth
    # v_hat_x = y[:, 0::3]
    # v_hat_y = y[:, 1::3]
    # v_hat_z = y[:, 2::3]

    error_list = []
    for s in range(len(v_hat)):

        # i=args.sample_index

        err_v = v_hat[s,:]-y[s,:]
        err = np.linalg.norm(err_v)
        error_list.append(err)
        # print(f"Error of sample {s}: {err}")

    # Initial position, last in the input trajectory
    i = args.sample_index
    p0_x = pos_in_tx[i,-1]
    p0_y = pos_in_ty[i,-1]
    p0_z = pos_in_tz[i,-1]

    pos_hat_x=[]
    pos_hat_y=[]
    pos_hat_z=[]
    # Need to compute positions from estimated velocities and dt
    for j in range(len(v_hat_x[i])):
        p0_x += dt*v_hat_x[i,j]
        pos_hat_x.append(p0_x)

        p0_y +=  dt*v_hat_y[i,j]
        pos_hat_y.append(p0_y)

        p0_z +=  dt*v_hat_z[i,j]
        pos_hat_z.append(p0_z)

    # In[Plot]
    # Need to plot 3D trajectpries
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pos_dataset[:,1], pos_dataset[:,2], pos_dataset[:,3], c='blue', s=1, label='original_trajectory')
    ax1.scatter(pos_in_tx[i], pos_in_ty[i], pos_in_tz[i], c='black', s=10, label=f"input positions: sequence{i}")
    ax1.scatter(pos_hat_x, pos_hat_y, pos_hat_z, c='red',s=10, label=f"predicted positions: sequence{i}")

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.legend()
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(error_list)
    ax2.set_xlabel("Index of trajectory sample")
    ax2.set_ylabel("Norm of error")
    ax2.set_title("Error with respect to ground truth")
    plt.show()