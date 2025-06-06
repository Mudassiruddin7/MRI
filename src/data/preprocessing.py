from preprocessing_utils import save_preprocessed_data_to_tfrecord
import argparse
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MRI images and masks and create TFRecords.")
    
    parser.add_argument("--base_dir", required=True, help="GCS path to the directory containing base directory.")
    parser.add_argument("--output_dir", required=True, help="GCS path to store the TFRecord files.")
    parser.add_argument("--image_height", type=int, default=512, help="Target image height.")
    parser.add_argument("--image_width", type=int, default=512, help="Target image width.")

    args = parser.parse_args()

    save_preprocessed_data_to_tfrecord( base_dir= args.base_dir, output_dir= args.output_dir,target_shape= (args.image_height, args.image_width))

if __name__ == "__main__":
    main()