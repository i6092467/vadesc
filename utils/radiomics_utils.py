"""
Utility functions for extracting radiomics features.
"""
import os

import shutil

import numpy as np

import cv2

import logging

import progressbar

from radiomics import featureextractor


def extract_radiomics_features(data_file, masks, verbose=1):
    # Set logging for the radiomics library
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    # Load images and segmentation masks
    images = np.load(file=data_file, allow_pickle=True)

    print(images.shape, masks.shape)

    assert images.shape == masks.shape

    # Create a temporary directory for images and masks
    if os.path.exists('./radiomics_features_temp'):
        shutil.rmtree('./radiomics_features_temp')
    else:
        os.makedirs('./radiomics_features_temp')

    n_images = images.shape[0]

    if verbose:
        print('Extracting radiomics features...')
        bar = progressbar.ProgressBar(maxval=n_images)
        bar.start()

    # Feature extraction by PyRadiomics
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()

    radiomics_features = None

    for i in range(n_images):
        # Create a directory for each image
        os.makedirs('./radiomics_features_temp/' + str(i))
        imageName = './radiomics_features_temp/' + str(i) + '/image.png'
        maskName = './radiomics_features_temp/' + str(i) + '/mask.png'
        cv2.imwrite(filename=imageName, img=images[i, 0])
        cv2.imwrite(filename=maskName, img=masks[i, 0])
        # Provide mask and image files to the extractor
        result = extractor.execute(imageFilepath=imageName, maskFilepath=maskName)
        result_features = [val for key, val in result.items() if 'original_' in key and 'diagnostics_' not in key]
        result_features = [float(r) for r in result_features]
        if radiomics_features is None:
            radiomics_features = np.zeros((n_images, len(result_features)))
        radiomics_features[i] = result_features

        if verbose > 0:
            bar.update(i)
    shutil.rmtree('./radiomics_features_temp')

    return radiomics_features
