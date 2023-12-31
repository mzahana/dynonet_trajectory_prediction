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
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trains a Dynonet network using trajectory dataset.")
    parser.add_argument("input_file", help="Path to the .npz file.")
    parser.add_argument("-it", "--iterations", type=int, default=10000, help="Number of training iterations")
    args = parser.parse_args()
    print(f"Traingin data: {args.input_file}")
    print(f"Number of iterations: {args.iterations}")

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr = 1e-3
    num_iter = args.iterations
    msg_freq = 100
    # n_fit = 500

    # Extract data
    try:
        loaded_data = np.load(args.input_file)
    except Exception as e:
        print("Error in reading dataset: {}".format(e))
        exit(1)

    # pos_dataset = loaded_data["pos_dataset"]
    # dt = loaded_data["dt"]

    u = loaded_data["vel_in"]
    y = loaded_data["vel_out"]

    # Number of trajectories points (samples/sec * points)
    u_N = u.shape[1]
    y_N = y.shape[1]

    # Number of training points
    num_pts = u.shape[0]

    print("number of training points:", num_pts)

    # Model blocks
    # F1_factor=1
    # G1 = MimoLinearDynamicalOperator(u_N, u_N*F1_factor, n_b=2, n_a=2, n_k=1)
    # # Static sandwitched non-linearity
    # F1 = MimoStaticNonLinearity(u_N*F1_factor, y_N*F1_factor, activation='tanh')
    # G2 = MimoLinearDynamicalOperator(y_N*F1_factor, y_N, n_b=2, n_a=2, n_k=0)

    factor=5
    F1 = MimoStaticNonLinearity(u_N, y_N*factor, activation='tanh')
    F2 = MimoStaticNonLinearity(y_N*factor, y_N, activation='tanh')
    G1 = MimoLinearDynamicalOperator(y_N, y_N, n_b=2, n_a=3, n_k=0)
    F3 = MimoStaticNonLinearity(y_N, y_N*factor, activation='tanh')
    F4 = MimoStaticNonLinearity(y_N*factor, y_N, activation='tanh')

    # Load identified model parameters
    # model_name = 'drone_trajecrory_model'
    # model_folder = os.path.join("models", model_name)
    # G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pkl")))
    # F1.load_state_dict(torch.load(os.path.join(model_folder, "F1.pkl")))
    # G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pkl")))

    # Model structure
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

    # In[Optimizer]
    # optimizer = torch.optim.Adam([
    #     {'params': G1.parameters(), 'lr': lr},
    #     {'params': F1.parameters(), 'lr': lr},
    # ], lr=lr)

    optimizer = torch.optim.Adam([
        {'params': F1.parameters(), 'lr': lr},
        {'params': F2.parameters(), 'lr': lr},
        {'params': G1.parameters(), 'lr': lr},
        {'params': F3.parameters(), 'lr': lr},
        {'params': F4.parameters(), 'lr': lr},
    ], lr=lr)

    # In[Prepare tensors]
    u_fit_torch = torch.tensor(u[None, :, :])
    y_fit_torch = torch.tensor(y[None, :, :])

    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        y_hat = model(u_fit_torch)

        err_fit = y_fit_torch - y_hat
        loss = torch.mean(err_fit ** 2)

        LOSS.append(loss.item())
        if itr % msg_freq == 0:
            print(f'Iter {itr} | Fit Loss {loss:.6f}')

        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    if (train_time< 60.):
        print(f"\nTrain time: {train_time:.2f} seconds")
    elif (train_time < (60.*60.)):
        print(f"\nTrain time: {train_time/60.:.2f} minutes")
    else:
        print(f"\nTrain time: {train_time/3600.:.2f} hours")

    print("Training iterations:", num_iter)
    print("Last loss:", loss.item())

    # In[Save model]
    model_name = 'drone_trajecrory_model'
    if model_name is not None:
        model_folder = os.path.join("models", model_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
        # torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
        # torch.save(G2.state_dict(), os.path.join(model_folder, "G2.pkl"))

        # torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
        # torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
        # torch.save(F2.state_dict(), os.path.join(model_folder, "F2.pkl"))

        torch.save(F1.state_dict(), os.path.join(model_folder, "F1.pkl"))
        torch.save(F2.state_dict(), os.path.join(model_folder, "F2.pkl"))
        torch.save(G1.state_dict(), os.path.join(model_folder, "G1.pkl"))
        torch.save(F3.state_dict(), os.path.join(model_folder, "F3.pkl"))
        torch.save(F4.state_dict(), os.path.join(model_folder, "F4.pkl"))

    # In[Detach]
    y_hat_np = y_hat.detach().numpy()[0, :, 0]

    # In[Plot loss]
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.plot(LOSS)
    plt.show()
