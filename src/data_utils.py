"""
Reusable data-inspection utilities for the Diabetic Retinopathy project.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# File integrity checks

def check_missing_and_corrupted(df, images_dir, id_col="id_code", ext=".png"):

    missing_files = []
    corrupted_files = []

    for img_id in df[id_col]:
        img_path = os.path.join(images_dir, img_id + ext)

        if not os.path.exists(img_path):
            missing_files.append(img_id)
            continue

        try:
            img = Image.open(img_path)
            img.verify()  # checks file integrity without fully decoding pixel data
        except Exception:
            corrupted_files.append(img_id)

    return missing_files, corrupted_files



# Duplicate / leakage detection

def compute_hash(img_path):

    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates_within(df, images_dir, id_col="id_code", ext=".png"):
    
    hashes = {}
    duplicates = []

    for img_id in df[id_col]:
        img_path = os.path.join(images_dir, img_id + ext)
        h = compute_hash(img_path)

        if h in hashes:
            duplicates.append((img_id, hashes[h]))
        else:
            hashes[h] = img_id

    return hashes, duplicates


def find_cross_split_leaks(other_df, other_images_dir, reference_hashes,
                            id_col="id_code", ext=".png"):
    """
    Check whether any image in `other_df` (e.g. valid or test) is an exact
    duplicate of an image already seen in `reference_hashes` (e.g. train).
    """
    leaks = []
    for img_id in other_df[id_col]:
        img_path = os.path.join(other_images_dir, img_id + ext)
        h = compute_hash(img_path)
        if h in reference_hashes:
            leaks.append((img_id, reference_hashes[h]))
    return leaks


def remove_leaked_ids(df, leak_pairs_list, id_col="id_code"):
    """
    Given one or more lists of (other_id, train_id) leak pairs, remove the
    corresponding train_id rows from `df`.
    """
    leaked_ids = set()
    for leak_pairs in leak_pairs_list:
        for _, source_id in leak_pairs:
            leaked_ids.add(source_id)

    df_clean = df[~df[id_col].isin(leaked_ids)].reset_index(drop=True)
    return df_clean, leaked_ids


# Class distribution
def class_distribution_report(df, label_col="diagnosis"):
   
    counts = df[label_col].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(2)
    ratios = (counts.max() / counts).round(2)

    report = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
        "imbalance_ratio": ratios,
    })
    return report

"""Bar chart of class counts."""
def plot_class_distribution(df, label_col="diagnosis", title="Class Distribution",
                             save_path=None):
    
    counts = df[label_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_xlabel("Diagnosis Grade")
    ax.set_ylabel("Number of Images")
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 15, str(v), ha="center")

    save_figure(fig, save_path)
    plt.show()
    return fig

# Blur / sharpness quantification

def compute_blurriness(img_path): 
#Laplacian-variance sharpness score. Higher = sharper.
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_blur_data(df, img_dir, split_name, id_col="id_code", label_col="diagnosis",
                   ext=".png", n_samples=300, random_state=42):
    """
    Computes blur score for a sample of images in a split 
    """
    sample_df = df.sample(min(n_samples, len(df)), random_state=random_state)

    records = []
    for _, row in sample_df.iterrows():
        img_id = str(row[id_col])
        img_path = os.path.join(img_dir, f"{img_id}{ext}")

        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id)

        blur = compute_blurriness(img_path)
        if blur is not None:
            records.append({
                "id_code": img_id,
                "diagnosis": row.get(label_col, None),
                "blur": blur,
                "split": split_name,
            })

    return pd.DataFrame(records)



def show_samples_per_class(df, images_dir, id_col="id_code", label_col="diagnosis",
                            ext=".png", n_per_class=6, random_state=42, save_path=None):
    
    grades = sorted(df[label_col].unique())
    fig, axes = plt.subplots(len(grades), n_per_class,
                              figsize=(n_per_class * 3, len(grades) * 3))

    for row, grade in enumerate(grades):
        sample_ids = df[df[label_col] == grade][id_col].sample(
            n_per_class, random_state=random_state
        ).values
        for col, img_id in enumerate(sample_ids):
            img_path = os.path.join(images_dir, img_id + ext)
            img = Image.open(img_path)
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Grade {grade}", fontsize=12)
            ax.set_title(img_id, fontsize=8)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()
    return fig



# Generic figure saving
def save_figure(fig, save_path, dpi=150):
    
    if save_path is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")