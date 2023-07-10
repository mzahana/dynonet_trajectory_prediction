import numpy as np

if __name__ == '__main__':
    # list of datasets
    ds = []
    # Extract data
    try:
        f_path="dynonet_datasets/indoor_forward_3_davis.npz"
        ds.append(np.load(f_path))
        print("Read:", f_path)
    except Exception as e:
        print("Error in reading dataset: {}".format(e))
        exit(1)

    try:
        f_path = "dynonet_datasets/indoor_forward_5_davis.npz"
        ds.append(np.load(f_path))
        print("Read:", f_path)
    except Exception as e:
        print("Error in reading dataset: {}".format(e))
        exit(1)

    # Merge datasets
    if len(ds)==0:
        print("ERROR: No datasets to merge")
        exit(1)

    # Fields: pos_dataset, pos_in, pos_out, vel_in, vel_out, dt
    # Fields to merge: pos_dataset, pos_in, pos_out, vel_in, vel_out
    dt = ds[0]["dt"]
    merged_pos_dataset = ds[0]["pos_dataset"]
    merged_pos_in = ds[0]["pos_in"]
    merged_pos_out = ds[0]["pos_out"]
    merged_vel_in = ds[0]["vel_in"]
    merged_vel_out = ds[0]["vel_out"]

    for i in range(len(ds)):
        merged_pos_dataset  = np.concatenate((merged_pos_dataset, ds[i]["pos_dataset"]), axis=0)
        merged_pos_in  = np.concatenate((merged_pos_in, ds[i]["pos_in"]), axis=0)
        merged_pos_out  = np.concatenate((merged_pos_out, ds[i]["pos_out"]), axis=0)
        merged_vel_in  = np.concatenate((merged_vel_in, ds[i]["vel_in"]), axis=0)
        merged_vel_out  = np.concatenate((merged_vel_out, ds[i]["vel_out"]), axis=0)

    print("merged_pos_dataset shape:", merged_pos_dataset.shape)
    print("merged_pos_in shape:", merged_pos_in.shape)
    print("merged_pos_out shape:", merged_pos_out.shape)
    print("merged_vel_in shape:", merged_vel_in.shape)
    print("merged_vel_out shape:", merged_vel_out.shape)


    # Save to a file
    output_file = "dynonet_datasets/merged_traj_datasets.npz"
    np.savez(output_file,
                pos_dataset=merged_pos_dataset,
                pos_in=merged_pos_in, 
                pos_out=merged_pos_out,
                vel_in=merged_vel_in,
                vel_out=merged_vel_out,
                dt=dt
            )
    print("Arrays saved to", output_file) 


        