import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def PlotTrajectory(original_file: str,resampled_file: str):

    df1 = pd.read_csv(original_file,delimiter=" ")
    df2 = pd.read_csv(resampled_file,delimiter=",")
    fig = plt.figure(figsize=(10, 5))

    print(df2)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(df1['tx'], df1['ty'], df1['tz'])

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(df2['tx'], df2['ty'], df2['tz'])

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()


def main():
    return

if __name__== "__main__":
    resampled_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_5_snapdragon_with_gt.txt')
    original_filename= ('/home/asmbatati/Documents/outdoor_forward_5_snapdragon_with_gt/groundtruth.txt')
    PlotTrajectory(original_filename,resampled_filename)     

