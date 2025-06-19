import argparse
import yaml
from os.path import exists, join
import xarray as xr
import numpy as np
from mlmicrophysics.data import split_staggered_variable
from mlmicrophysics.data import load_cam_output, unstagger_vertical, convert_to_dataframe
from mlmicrophysics.data import calc_pressure_field, calc_temperature, add_index_coords
from glob import glob
from dask.distributed import Client, LocalCluster, wait
import traceback
from os import makedirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration yaml file")
    parser.add_argument("-p", "--proc", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    if not exists(config["out_path"]):
        makedirs(config["out_path"])

    # Get all filenames for processing (E3SM 12.3 data - all 7 files)
    filenames = sorted(glob(join(config["model_path"],
                                 config["model_file_start"] + "*" + config["model_file_end"])))
    
    # PROCESSING: Process all files sequentially for memory stability
    print(f"Processing all {len(filenames)} E3SM 12.3 files:")
    for f in filenames:
        print(f"  - {f}")
    
    if "dt" not in config.keys():
        config["dt"] = 1800
    
    # Run sequentially for stability
    for i, filename in enumerate(filenames):
        print(f"Processing file {i+1}/{len(filenames)}: {filename}")
        process_cesm_file_subset(filename,
                                 staggered_variables=config["staggered_variables"],
                                 out_variables=config["out_variables"],
                                 subset_variable=config["subset_variable"],
                                 subset_threshold=config["subset_threshold"],
                                 out_path=config["out_path"],
                                 out_start=config["out_start"],
                                 out_format=config["out_format"],
                                 dt=config["dt"])
    return


def process_cesm_file_subset(filename, staggered_variables=None, time_var="time", out_variables=None,
                             subset_variable=None, subset_threshold=None, out_path="./",
                             out_start="cam_mp_data", out_format="csv", dt=1800):
    print("Processing " + filename)
    model_ds = xr.open_dataset(filename, decode_times=False, chunks={})

    times = model_ds[time_var]
    for time in times:
        time_hours = int(time * 24)
        print(f"  Processing time step: {time_hours}")
        time_df = model_ds[out_variables].sel(**{time_var: time}).to_dataframe()
        if type(subset_variable) == list:
            valid = np.zeros(time_df.shape[0], dtype=bool)
            for s, sv in enumerate(subset_variable):
                valid[time_df[sv] >= subset_threshold[s]] = True
        else:
            valid = time_df[subset_variable] >= subset_threshold
        time_sub_df = time_df.loc[valid].reset_index()
        del time_df
        time_sub_df.rename(columns={"P3_qc_in_TAU": "QC_TAU_in",
                                    "P3_nc_in_TAU": "NC_TAU_in",
                                    "P3_qr_in_TAU": "QR_TAU_in",
                                    "P3_nr_in_TAU": "NR_TAU_in",
                                    "P3_qc_out_TAU": "QC_TAU_out",
                                    "P3_nc_out_TAU": "NC_TAU_out",
                                    "P3_qr_out_TAU": "QR_TAU_out",
                                    "P3_nr_out_TAU": "NR_TAU_out",
                                    "P3_mu_c": "PGAM",
                                    "P3_lamc": "LAMC",
                                    "P3_lamr": "LAMR",
                                    "P3_nr": "N0R",
                                    "P3_qctend_TAU_raw": "qctend_TAU",
                                    "P3_nctend_TAU_raw": "nctend_TAU",
                                    "P3_qrtend_TAU_raw": "qrtend_TAU",
                                    "P3_nrtend_TAU_raw": "nrtend_TAU",
                                    }, inplace=True)
        if out_format == "csv":
            time_sub_df.to_csv(join(out_path, "{0}_{1:06d}.csv".format(out_start, time_hours)),
                               index_label="Index")
        elif out_format == "parquet":
            time_sub_df.to_parquet(join(out_path, "{0}_{1:06d}.parquet".format(out_start, time_hours)))
    model_ds.close()
    del model_ds
    print(f"Completed processing {filename}")
    return


if __name__ == "__main__":
    main() 