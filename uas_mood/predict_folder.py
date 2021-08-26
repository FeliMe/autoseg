import argparse
from glob import glob
import os

import torch
from tqdm import tqdm

from uas_mood.train_patch_interpolation import LitModel
from uas_mood.utils.data_utils import process_scan, save_nii, write_txt
from uas_mood.utils.evaluation import samplewise_score


def predict_folder(input_dir, output_dir, mode, model_ckpt, verbose):
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read all input files to a list
    if input_dir[-1] != "/":
        input_dir += "/"
    files = glob(f"{input_dir}*.nii*")
    print(f"Found {len(files)} files in {input_dir}")

    # Load model
    model = LitModel.load_from_checkpoint(model_ckpt).to(device)

    if verbose:
        print("Model hyperparameters")
        print(model.hparams)

    # Create output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over files
    pbar = tqdm(files)
    for f in pbar:
        fname = f.split('/')[-1]

        # Target filename
        pbar.set_description(fname)

        # Load nii file
        x, affine = process_scan(f, model.args.img_size, False, return_affine=True)
        x = torch.from_numpy(x).to(device)

        # Run forward
        with torch.no_grad():
            pred = model.predict_volume(x, batch_size=8).cpu().numpy()

        # Save target
        if "pixel" in mode:
            t = os.path.join(output_dir, fname)
            save_nii(t, pred, affine, primary_axis=2, dtype="float32")
        if "sample" in mode:
            t = os.path.join(output_dir, fname + ".txt")
            pred = samplewise_score(pred)
            write_txt(t, str(pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-m", "--mode", type=str, nargs='+',
                        default=["pixel", "sample"],
                        help="can be either 'pixel' or 'sample'.")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    mode = args.mode
    model_ckpt = args.model_ckpt
    verbose = args.verbose

    if not isinstance(mode, list):
        mode = list(mode)

    predict_folder(input_dir, output_dir, mode, model_ckpt, verbose)
