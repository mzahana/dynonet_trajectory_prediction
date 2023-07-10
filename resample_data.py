import csv
import argparse
import numpy as np
import pandas as pd
import os



def processData(inp_filename: str, out_filename: str, dt: float):
    
    try:
        df = pd.read_csv(inp_filename, delimiter=" ")
    except Exception as e:
        print("Error {}".format(e))
        return
    
    # print("Number of original data samples = ", df.shape[0])

    try:
        # Find smallest time difference
        # Calculate the time differences between adjacent rows
        time_diffs = (df['timestamp'] - df['timestamp'].shift()).dropna()

        # Find the minimum time difference
        min_time_diff = time_diffs.min()
        max_time_diff = time_diffs.max()

        if dt < min_time_diff:
            print("Warning: requested sampling rate {} is < the minimim data sampling rate dt_data= {}".format(dt, min_time_diff))
            return
    except Exception as e:
        print("Error: {}".format(e))
        return
    
    # Resample
    try:
        df.index = pd.to_datetime(df['timestamp'], unit='s')
        dt_str=str(int(dt*1000*1000))+'us' #us is for micro-second, S is for seoconds,  L is for milli-second in Pandas
        df_resampled = df.resample(dt_str).mean().interpolate()
        
        time_diffs = (df_resampled['timestamp'] - df_resampled['timestamp'].shift()).dropna()
        
        re_min_time_diff = time_diffs.min()
        re_max_time_diff = time_diffs.max()
        re_avg_dt = time_diffs.mean()
    except Exception as e:
        print("Error: {}".format(e))
        return
    
    # print(df_resampled.tail)

    print("Number of samples after resampling: {}".format(df_resampled.shape[0]))
    print("Original data - Minimum dt = {}, Max dt= {}".format(min_time_diff, max_time_diff))
    print("Resampeld data - Minimum dt = {}, Max dt= {}, Average= {}".format(re_min_time_diff, re_max_time_diff,re_avg_dt))

    try:
        df_resampled.to_csv(out_filename, index=False)
        print("Output file is saved at {}".format(out_filename))
    except Exception as e:
        print("Error: {}".format(e))
    

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='CSV data reader')

    # Define the arguments
    parser.add_argument('input_file', help='Input file name')
    parser.add_argument('--output', '-o', help='Output file name', default='output.txt')
    parser.add_argument('--sampling_time', '-t', help='Sampling time', default=0.05)
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Parse the arguments
    args = parser.parse_args()

    # processData(args.input_file, args.output, float(args.sampling_time))
    processData()
    

if __name__== "__main__":
    inp_filename = ('original_datasets/indoor_forward_3_davis_with_gt/groundtruth.txt')

    output_dir = "resampled_100ms_dataset"

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

    out_filename = (directory_path+'/indoor_forward_3_davis_with_gt.txt')
    dt = 0.1
    processData(inp_filename, out_filename, dt)
    
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_6_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_6_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_5_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_5_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_7_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_7_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_9_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_9_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_10_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_10_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/outdoor_forward_1_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_1_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/outdoor_forward_3_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_3_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/outdoor_forward_5_davis_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_5_davis_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)


    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_3_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_3_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_6_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_6_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_5_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_5_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_7_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_7_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_9_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_9_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/indoor_forward_10_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/indoor_forward_10_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/outdoor_forward_1_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_1_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/outdoor_forward_3_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_3_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # inp_filename = ('/home/asmbatati/Documents/outdoor_forward_5_snapdragon_with_gt/groundtruth.txt')
    # out_filename = ('/home/asmbatati/Documents/modified_trajectories_datasets/outdoor_forward_5_snapdragon_with_gt.txt')
    # dt = 0.01
    # processData(inp_filename, out_filename, dt)
    # # main()

