import argparse
from glob import glob

from uas_mood.train_patch_interpolation import LitModel

def predict_folder(input_dir, output_dir, mode, data, model_ckpt):
    # Read all input files to a list
    files = glob(f"{input_dir}*.nii*")

    # Load model
    model = LitModel.load_from_checkpoint(model_ckpt)

    import IPython ; IPython.embed() ; exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("--mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.")
    parser.add_argument("--model_ckpt", type=str, required=True)

    parser.add_argument("-d","--data", type=str, help="can be either 'brain' or 'abdom'.", required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    mode = args.mode
    data = args.data
    model_ckpt = args.model_ckpt

    predict_folder(input_dir, output_dir, mode, data, model_ckpt)
