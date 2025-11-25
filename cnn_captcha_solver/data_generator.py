import os
import shutil
import random
from typing import List, Tuple

import pandas as pd
import cv2

from .segmenter import Segmenter


class DataGenerator:
    """
    Data generator that takes a directory of CAPTCHA images and splits them
    into directories for training and testing sets.

    Parameters
    ----------
    src_dir : str
        Path to directory that contains CAPTCHA images. Image files must be
        named with their CAPTCHA text (e.g. "A1b2.png").
    random_seed : int | None, default=None
        Integer seed to set random.seed() for reproducible splits.
    train_size : float, default=0.75
        Fraction of the dataset to use for training (0.0 - 1.0).

    Attributes
    ----------
    train_img_paths : list[str]
        Paths to CAPTCHA images in the training set.
    test_img_paths : list[str]
        Paths to CAPTCHA images in the test set.
    label_dict : dict[str, int]
        Mapping from character to integer label.
    segmenter : Segmenter
        Instance of the Segmenter class used to segment captchas.
    """

    def __init__(self, src_dir: str, train_size: float = 0.75, random_seed: int | None = None):
        if random_seed is not None:
            random.seed(random_seed)

        # collect image paths
        img_paths = [
            os.path.join(src_dir, fname)
            for fname in os.listdir(src_dir)
            if fname.lower().endswith((".jpg", ".png"))
        ]

        random.shuffle(img_paths)
        split_index = int(len(img_paths) * train_size)
        self.train_img_paths: List[str] = img_paths[:split_index]
        self.test_img_paths: List[str] = img_paths[split_index:]

        # generate list of all characters from filenames (safe split)
        all_chars: List[str] = []
        for img_path in img_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            all_chars.extend(list(base))

        unique_chars = sorted(set(all_chars))
        # assign integer label to each unique character
        self.label_dict = {char: i for i, char in enumerate(unique_chars)}

        # instance of Segmenter
        self.segmenter = Segmenter()

    def extract_train_set(self, target_dir: str, train_annotation_file: str = "train_annotations.csv"):
        """
        Uses the Segmenter to extract individual characters from the training
        CAPTCHA images and save them (and an annotations CSV) to target_dir.

        Parameters
        ----------
        target_dir : str
            Path for a new directory (or empty existing directory) to contain
            the extracted character images.
        train_annotation_file : str
            Filename for the annotations CSV inside target_dir.
        """
        # Ensure target_dir exists or create it
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        elif any(fname.lower().endswith((".png", ".jpg")) for fname in os.listdir(target_dir)):
            raise ValueError(
                "target_dir needs to be a new directory or an existing directory "
                "free of image files"
            )

        annotation_rows: List[dict] = []

        print(f"Segmenting chars from {len(self.train_img_paths)} images...")

        i = 0
        for img_path in self.train_img_paths:
            segmented_chars: List[Tuple] = self.segmenter.segment_chars(img_path)

            for char_img, label in segmented_chars:
                if i % 2000 == 0:
                    print(f"Working on char {i}...")

                target_img_fn = f"{str(i).zfill(6)}.png"
                target_img_path = os.path.join(target_dir, target_img_fn)
                try:
                    # write segmented character image to target directory
                    success = cv2.imwrite(target_img_path, char_img)
                    if not success:
                        print(f"Failed to write image to {target_img_path} (cv2.imwrite returned False).")
                        continue

                    # map char label to integer label
                    int_label = self.label_dict[label]
                    annotation_rows.append({"filename": target_img_fn, "label": int_label})
                    i += 1
                except cv2.error as e:
                    print("\n*************************"
                          "\nFailed to write image with cv2.error:")
                    print(e)
                    continue
                except KeyError:
                    # unlikely: label not in label_dict
                    print(f"Warning: label '{label}' not found in label_dict; skipping.")
                    continue

        # save annotations CSV inside target_dir
        annotation_df = pd.DataFrame(annotation_rows)
        annotation_csv_path = os.path.join(target_dir, train_annotation_file)
        annotation_df.to_csv(annotation_csv_path, index=False)
        print(f"Done! Saved filename/numeric class label key to file: {annotation_csv_path}")

    def save_test_set(self, target_dir: str):
        """
        Copy over test set images to target directory. Test images remain
        unsegmented/unmodified to use as unseen evaluation data.
        """
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        elif any(fname.lower().endswith((".png", ".jpg")) for fname in os.listdir(target_dir)):
            raise ValueError(
                "target_dir needs to be a new directory or an existing directory "
                "free of image files"
            )

        print(f"Copying {len(self.test_img_paths)} images to {target_dir}...")

        for img_path in self.test_img_paths:
            target_path = os.path.join(target_dir, os.path.basename(img_path))
            shutil.copy(img_path, target_path)

        print(f"Copied {len(self.test_img_paths)} images to {target_dir}")

    def save_label_dict(self, target_path: str):
        """
        Save label dictionary of integer labels of alphanumeric characters
        within CAPTCHA images to a CSV at target_path.
        """
        label_list = list(self.label_dict.items())
        label_df = pd.DataFrame(label_list, columns=["char", "int"])
        label_df.to_csv(target_path, index=False)
