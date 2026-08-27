import os
 
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

def crop_black_border(img,threshold=9):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask=gray>threshold #considered as foreground
    coords=np.argwhere(mask)

    if coords.size==0:
        return img

    #bounding box enclosing the foreground
    y_min, x_min = coords.min(axis=0) #top-left corner
    y_max, x_max = coords.max(axis=0) #bottom-right corner

    cropped= img[y_min:y_max+1, x_min:x_max+1]
    return cropped

def pad_to_square(img):
   
    h, w = img.shape[:2]
    size = max(h, w)
    
    # compute padding 
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

def resize_image(img, target_size=224):

    resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    #BGR to lab 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result

def normalize_for_efficientnet(img):
    img= img.astype(np.float32)
    img = preprocess_input(img)
    return img

TARGET_SIZE = 224

def full_pipeline_preprocess(img_bgr, target_size=TARGET_SIZE, apply_enhancement=True):
    img = crop_black_border(img_bgr, threshold=9)
    img = pad_to_square(img)
    img = resize_image(img, target_size=target_size)
    if apply_enhancement:
        img = apply_clahe(img)          # expects BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    img = img.astype(np.float32)
    img = preprocess_input(img)          # no-op for EfficientNet
    return img

def _process_numpy(path_tensor, target_size=TARGET_SIZE, apply_enhancement=True):
        path = path_tensor.numpy().decode('utf-8')
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"Could not load image at path: {path}")
        return full_pipeline_preprocess(img_bgr, target_size=target_size,
                                     apply_enhancement=apply_enhancement)

def load_and_process(path_tensor, label_tensor, target_size=TARGET_SIZE):
    """
    tf.data-compatible wrapper around full_pipeline_preprocess. Uses
    tf.py_function to bridge OpenCV/NumPy code into the TensorFlow graph,
    then manually restores shape info (tf.py_function outputs are
    shapeless by default, which breaks downstream batching).
    """
    img = tf.py_function(
        func=lambda p: _process_numpy(p, target_size=target_size),
        inp=[path_tensor],
        Tout=tf.float32,
    )
    img.set_shape([target_size, target_size, 3])
    label = tf.cast(label_tensor, tf.int32)
    return img, label

def build_paths_labels(df, images_dir, id_col="id_code", label_col="diagnosis", ext=".png"):
    paths = [os.path.join(images_dir, f"{img_id}{ext}") for img_id in df[id_col]]
    labels = df[label_col].values
    return paths, labels

def build_oversampled_paths_labels(paths, labels, oversample_map):
    """
   increases EXPOSURE, not real diversity.
   Intended to be used as an alternative to class_weight, not alongside it.
    """
    paths = np.array(paths)
    labels = np.array(labels)
 
    out_paths, out_labels = [], []
    for cls in np.unique(labels):
        mult = oversample_map.get(int(cls), 1)
        cls_paths = paths[labels == cls]
        cls_labels = labels[labels == cls]
        out_paths.extend(np.tile(cls_paths, mult).tolist())
        out_labels.extend(np.tile(cls_labels, mult).tolist())
 
    return out_paths, out_labels


def build_augmentation():
    """
    Train-only augmentation. Deliberately conservative: flips and mild
    rotation only, since retinas have no fixed orientation. No color/
    brightness augmentation — EDA flagged both blur and color as
    potentially carrying real diagnostic signal, so randomizing them
    risks corrupting information the model needs.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
    ])

def make_dataset(paths, labels, training=False, batch_size=16, target_size=TARGET_SIZE):
    """
    Builds the full tf.data pipeline: load+preprocess -> cache -> (shuffle)
    -> batch -> (augment) -> prefetch.
 
    .cache() sits right after preprocessing (before shuffle/augment) because
    the cv2 pipeline is deterministic and CPU-heavy — caching it means the
    expensive work runs once on epoch 1, not every epoch. Caching after
    augmentation would freeze "random" augmentations into a fixed version,
    defeating their purpose.

    """
    AUTOTUNE = tf.data.AUTOTUNE
 
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: load_and_process(p, l, target_size=target_size),
                num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
 
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
 
    ds = ds.batch(batch_size)
 
    if training:
        augment = build_augmentation()
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
 
    ds = ds.prefetch(AUTOTUNE)
    return ds


# Class imbalance handling

def compute_class_weights(labels):
   
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))