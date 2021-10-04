"""
Utility functions for CT scan preprocessing.
"""
import numpy as np
import pandas as pd
import os
import glob
import cv2
import progressbar
import re
from PIL import Image, ImageOps

# Libraries for DICOM data handling
import pydicom
import pydicom_seg

LUNG1_N_PATIENTS = 422
RADIOGENOMICS_N_PATIENTS = 96

IGNORED_PATIENTS = np.unique([61, 179, 251, 352])
IGNORED_RADIOGENOMICS_PATIENTS = [7, 20, 21, 24, 36, 57, 74, 82, 87]
IGNORED_LUNG3_PATIENTS = [12, 13, 16, 24, 26, 27, 28, 34, 37, 38, 40, 44, 53, 56, 57, 63, 64, 66, 68, 72]
IGNORED_BASEL_PATIENTS = [3, 4, 5, 19, 32, 38, 41, 70, 76, 88, 107, 116, 119, 136, 153, 160, 164, 167, 178, 183,
                          199, 226, 237, 298, 302, 304, 306, 307, 310, 318, 337, 339, 347, 382, 385]

REFERENCE_SLICE_THICKNESS = 3.0

DAYS = [f'day{i}/' for i in np.arange(2, 31)]


# Loads a CT scan
def load_scan(path, f1=True):
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    if f1:
        slices = [s for s in slices if 'SliceLocation' in s]
        slices.sort(key=lambda x: int(x.SliceLocation), reverse=True)
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
    pixel_spacing = slices[0].PixelSpacing
    return {'slices': slices, 'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing}


# Transforms DICOM data to a pixel array
def get_pixels(scans, returnList=False):
    if not returnList:
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)
        return np.array(image, dtype=np.int16)
    else:
        return [s.pixel_array for s in scans]


# Performs histogram equalisation
def histogram_equalization(image, n_bins=256):
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), n_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape), cdf


# Performs histogram equalisation on a batch of images
def equalise_histograms(images, n_bins=256):
    images_eq = np.copy(images)
    for i in range(images.shape[0]):
        img_eq, cdf = histogram_equalization(images[i], n_bins=n_bins)
        images_eq[i, :, :] = img_eq
    return images_eq


# Performs minmax normalisation on a batch of images
def normalise_images(images):
    images_n = np.zeros_like(images)
    for i in range(images.shape[0]):
        img_n = cv2.normalize(images[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        images_n[i, :, :] = img_n
    return images_n


# Downscales an image batch to the specified size
def downscale_images(images, desired_size):
    downscaled = np.zeros((images.shape[0], desired_size[0], desired_size[1]))
    for j in range(images.shape[0]):
        downscaled[j] = cv2.resize(images[j], dsize=desired_size, interpolation=cv2.INTER_CUBIC)
    return downscaled


# Crops a CT slice around lungs, lung segmentation is optional
def crop_image(image, lung_segmentation=None, p_min=0.15):
    if lung_segmentation is None:
        th, threshed = cv2.threshold(image, p_min, 1, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed1 = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
        morphed = cv2.morphologyEx(morphed1, cv2.MORPH_CLOSE, kernel)
        morphed_int = np.array(morphed, dtype=np.uint8)
        cnts = cv2.findContours(morphed_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        dst = image[y:y + h, x:x + w]
    else:
        cnts = \
        cv2.findContours(np.array(lung_segmentation, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = np.concatenate((sorted(cnts, key=cv2.contourArea)[0], sorted(cnts, key=cv2.contourArea)[1]), axis=0)
        x, y, w, h = cv2.boundingRect(cnt)
        dst = image[y:y + h, x:x + w]
    return dst


# Rescales the image to the specified size
def resize_image(image, desired_size):
    img = Image.fromarray(image)
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    im = img.resize(new_size, Image.ANTIALIAS)
    new_im = ImageOps.expand(im, padding, fill=0)
    new_im = np.array(new_im)
    return new_im


# Crops CT scans and subsequently performs histogram equalisation
def crop_equalize_images(images, shape, n_bins, lung_segmentations=None, p_min=0.15):
    images_n = np.zeros([len(images), shape, shape])
    for i in range(images.shape[0]):
        if lung_segmentations is None:
            img = crop_image(images[i], p_min=p_min)
        else:
            img = crop_image(images[i], lung_segmentation=lung_segmentations[i], p_min=p_min)
        img, cdf = histogram_equalization(img, n_bins=n_bins)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        img = resize_image(img, shape)
        images_n[i, :, :] = img
    return images_n


def load_lung1_images_max_tumour_volume_ave(lung1_dir, n_slices, dsize, verbose=1):
    """
    Loads Lung1 dataset, takes an average 15 mm around the slice with the maximum transversal tumour area.
        https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics
    """
    assert n_slices % 2 == 1

    lung1_best_slices = np.zeros((LUNG1_N_PATIENTS, 1, dsize[0], dsize[1]))
    lung1_tumor_volumes = np.zeros((LUNG1_N_PATIENTS, 6))
    lung1_lung_volumes = np.zeros((LUNG1_N_PATIENTS,))
    lung1_lung_volumes_tot = np.zeros((LUNG1_N_PATIENTS,))
    lung1_best_slice_segmentations = np.zeros((LUNG1_N_PATIENTS, 1, dsize[0], dsize[1]))

    if verbose > 0:
        print('Loading CT data:')
        print()

    if os.path.exists('../datasets/nsclc_lung/lung1_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
            dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading from a pre-saved file...')
        lung1_best_slices = np.load(
            file='../datasets/nsclc_lung/lung1_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=LUNG1_N_PATIENTS)
            bar.start()

        for i in range(1, LUNG1_N_PATIENTS + 1):
            patient_dir = os.path.join(lung1_dir, 'LUNG1-' + "{:03d}".format(i))
            patient_dir = os.path.join(patient_dir, os.listdir(patient_dir)[0])
            patient_modalities = os.listdir(patient_dir)
            list.sort(patient_modalities)

            seg_modalities = [f for f in patient_modalities if re.search('Segmentation', f)]

            patient_seg_dir = None
            if len(seg_modalities) > 0:
                patient_seg_dir = os.path.join(patient_dir, seg_modalities[0])
            elif verbose > 0:
                print('WARNING: No segmentation for patient ' + str(i))

            patient_ct_dir = os.path.join(patient_dir, patient_modalities[0])
            results_dict = load_scan(patient_ct_dir)
            patient_ct_slices = results_dict['slices']
            slice_thickness = results_dict['slice_thickness']
            pixel_spacing = results_dict['pixel_spacing']
            n_slices_scaled = int(REFERENCE_SLICE_THICKNESS / slice_thickness * n_slices)
            patient_ct_pix = get_pixels(patient_ct_slices)
            patient_ct_pix_d = downscale_images(patient_ct_pix, (dsize[0], dsize[1]))

            if patient_seg_dir is not None:
                lung_seg_dcm = pydicom.dcmread(patient_seg_dir + str('/1-1.dcm'))
                seg_reader = pydicom_seg.SegmentReader()
                seg_result = seg_reader.read(lung_seg_dcm)
                seg_infos = seg_result.segment_infos
                n_segments = len(seg_infos)
                lung_left_seg = None
                lung_right_seg = None
                lung_tot_seg = None
                neoplasm_seg = None
                for s in range(1, n_segments + 1):
                    s_info = seg_infos[s]
                    if re.search('Neoplasm', str(s_info)):
                        neoplasm_seg = np.flip(seg_result._segment_data[s], 0)
                    elif re.search('Lung-Left', str(s_info)):
                        lung_left_seg = np.flip(seg_result._segment_data[s], 0)
                    elif re.search('Lung-Right', str(s_info)):
                        lung_right_seg = np.flip(seg_result._segment_data[s], 0)
                    elif re.search('Lungs-Total', str(s_info)):
                        lung_tot_seg = np.flip(seg_result._segment_data[s], 0)
                if neoplasm_seg is None and verbose > 0:
                    print('WARNING: No neoplasm segment for patient ' + str(i))
                if (lung_left_seg is None and lung_right_seg is None and lung_tot_seg is None) and verbose > 0:
                    print('WARNING: No lung segment for patient ' + str(i))

                tumour_vols = np.sum(neoplasm_seg, axis=(1, 2))
                tumour_vols_mm = np.sum(neoplasm_seg, axis=(1, 2)) * pixel_spacing[0] * pixel_spacing[
                    1] * slice_thickness
                lung_vols = None
                # lung_vols_mm = None
                if lung_left_seg is not None and lung_right_seg is not None:
                    lung_vols = np.sum(lung_left_seg, axis=(1, 2)) + np.sum(lung_right_seg, axis=(1, 2))
                elif lung_tot_seg is not None:
                    lung_vols = np.sum(lung_tot_seg, axis=(1, 2))
                best_slice_ind = np.argmax(tumour_vols)
                range_slices = np.arange((best_slice_ind - (n_slices_scaled - 1) // 2),
                                         (best_slice_ind + (n_slices_scaled - 1) // 2 + 1))
                if len(range_slices) > 0:
                    range_slices = range_slices[np.round(np.linspace(0, len(range_slices) - 1, n_slices)).astype(int)]

                if len(range_slices) == 0 or range_slices[0] >= patient_ct_pix.shape[0]:
                    best_slices = patient_ct_pix_d[0:2]
                    lung1_tumor_volumes[i - 1, 0] = np.sum(tumour_vols[0:2])
                    lung1_tumor_volumes[i - 1, 1] = np.sum(tumour_vols_mm[0:2])
                    if lung_vols is not None:
                        lung1_lung_volumes[i - 1] = np.sum(lung_vols[0:2])
                else:
                    best_slices = patient_ct_pix_d[range_slices]
                    lung1_tumor_volumes[i - 1, 0] = np.sum(tumour_vols[range_slices])
                    lung1_tumor_volumes[i - 1, 1] = np.sum(tumour_vols_mm[range_slices])
                    if lung_vols is not None:
                        lung1_lung_volumes[i - 1] = np.sum(lung_vols[range_slices])

                if lung_vols is not None:
                    lung1_lung_volumes_tot[i - 1] = np.sum(lung_vols)

                lung1_tumor_volumes[i - 1, 3] = np.sum(tumour_vols)
                lung1_tumor_volumes[i - 1, 4] = np.sum(tumour_vols_mm)

                lung1_best_slices[i - 1, 0, :, :] = np.mean(best_slices, axis=0)

                lung1_best_slice_segmentations[i - 1, 0, :, :] = (downscale_images(np.expand_dims(
                    neoplasm_seg[np.argmax(tumour_vols)], 0), (dsize[0], dsize[1]))[0] > 0) * 1.
            if verbose > 0:
                bar.update(i - 1)

    lung1_best_slices = lung1_best_slices.astype('float32')

    lung1_tumor_volumes[:, 2] = lung1_tumor_volumes[:, 0] / lung1_lung_volumes
    lung1_tumor_volumes[:, 5] = lung1_tumor_volumes[:, 3] / lung1_lung_volumes_tot

    if not os.path.exists(
            '../datasets/nsclc_lung/lung1_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                    dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Saving as a file...')

        np.save(file='../datasets/nsclc_lung/lung1_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
            dsize[1]) + str('.npy'), arr=lung1_best_slices, allow_pickle=True)
        np.save(file='../datasets/nsclc_lung/lung1_tumor_volumes.npy', arr=lung1_tumor_volumes, allow_pickle=True)
        np.save(file='../datasets/nsclc_lung/lung1_segmentations_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
            dsize[1]) + str('.npy'), arr=lung1_best_slice_segmentations, allow_pickle=True)

    lung1_best_slices = np.delete(lung1_best_slices, [127], axis=0)  # empty scan
    lung1_best_slices = np.expand_dims(lung1_best_slices, -1)
    return lung1_best_slices


def load_lung3_images_max_tumour_volume_ave(lung3_dir, n_slices, dsize, verbose=1):
    """
    Loads Lung3 dataset, takes an average 15 mm around the slice with the maximum transversal tumour area.
        https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics-Genomics
    """
    assert n_slices % 2 == 1

    master_table = pd.read_csv(os.path.join(lung3_dir, 'Lung3_master.csv'))
    lung3_n_patients = len(master_table['Case ID'].values)
    lung3_best_slices = np.zeros((lung3_n_patients, 1, dsize[0], dsize[1]))

    if verbose > 0:
        print('Loading CT data:')
        print()

    if os.path.exists('../datasets/nsclc_lung/lung3_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
            dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading from a pre-saved file...')
        lung3_best_slices = np.load(
            file='../datasets/nsclc_lung/lung3_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=lung3_n_patients)
            bar.start()

        for i in range(lung3_n_patients):
            patient_ct_dir = os.path.join(lung3_dir, master_table['CT directory'].values[i])
            results_dict = load_scan(patient_ct_dir)
            patient_ct_slices = results_dict['slices']
            slice_thickness = results_dict['slice_thickness']
            n_slices_scaled = int(REFERENCE_SLICE_THICKNESS / slice_thickness * n_slices)
            patient_ct_pix = get_pixels(patient_ct_slices)
            patient_ct_pix_d = downscale_images(patient_ct_pix, (dsize[0], dsize[1]))

            best_slice_ind = master_table['Tumor slice index'].values[i]
            range_slices = np.arange((best_slice_ind - (n_slices_scaled - 1) // 2),
                                     (best_slice_ind + (n_slices_scaled - 1) // 2 + 1))
            range_slices = range_slices[np.round(np.linspace(0, len(range_slices) - 1, n_slices)).astype(int)]
            if len(range_slices) == 0 or range_slices[0] >= patient_ct_pix.shape[0]:
                best_slices = patient_ct_pix_d[0:2]
            else:
                best_slices = patient_ct_pix_d[range_slices]
            lung3_best_slices[i, 0, :, :] = np.mean(best_slices, axis=0)

            if verbose > 0:
                bar.update(i)

    lung3_best_slices = lung3_best_slices.astype('float32')

    if not os.path.exists(
            '../datasets/nsclc_lung/lung3_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                    dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Saving as a file...')
        np.save(file='../datasets/nsclc_lung/lung3_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
            dsize[1]) + str('.npy'), arr=lung3_best_slices, allow_pickle=True)

    lung3_best_slices = np.expand_dims(lung3_best_slices, -1)
    return lung3_best_slices


def load_radiogenomics_images_max_tumour_volume_ave(radiogenomics_dir, n_slices, dsize, verbose=1):
    """
    Loads a subset of the NSCLC Radiogenomics dataset, takes an average 15 mm around the slice with the maximum
    transversal tumour area.
        https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics
    """
    assert n_slices % 2 == 1

    radiogenomics_best_slices = np.zeros((RADIOGENOMICS_N_PATIENTS, 1, dsize[0], dsize[1]))
    radiogenomics_best_slice_segmentations = np.zeros((RADIOGENOMICS_N_PATIENTS, 1, dsize[0], dsize[1]))

    if verbose > 0:
        print('Loading CT data:')
        print()

    if os.path.exists(
            '../datasets/nsclc_lung/radiogenomics_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading from a pre-saved file...')
        radiogenomics_best_slices = np.load(
            file='../datasets/nsclc_lung/radiogenomics_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=RADIOGENOMICS_N_PATIENTS)
            bar.start()

        # Load technical metadata
        meta_dat = pd.read_csv(os.path.join(radiogenomics_dir, 'metadata.csv'))
        # Find segmentations
        segmentation_directories = meta_dat['File Location'].values[
            meta_dat['Series Description'].values == '3D Slicer segmentation result']
        # Sanity check
        assert len(segmentation_directories) == RADIOGENOMICS_N_PATIENTS

        for i in range(0, RADIOGENOMICS_N_PATIENTS):
            # Construct segmentation directory
            patient_seg_dir = segmentation_directories[i]
            patient_seg_dir = os.path.join(radiogenomics_dir, patient_seg_dir.replace('./', ''))
            # Patient's data directory
            patient_dir = os.path.dirname(patient_seg_dir)
            patient_modalities = os.listdir(patient_dir)
            list.sort(patient_modalities)
            # CT data directory
            ct_modalities = [f for f in patient_modalities if not (re.search('segmentation result', f))]
            patient_ct_dir = os.path.join(patient_dir, ct_modalities[0])
            # Load CT
            results_dict = load_scan(patient_ct_dir)
            patient_ct_slices = results_dict['slices']
            slice_thickness = results_dict['slice_thickness']
            n_slices_scaled = int(REFERENCE_SLICE_THICKNESS / slice_thickness * n_slices)
            patient_ct_pix = get_pixels(patient_ct_slices)
            patient_ct_pix_d = downscale_images(patient_ct_pix, (dsize[0], dsize[1]))
            # Load segmentation
            lung_seg_dcm = pydicom.dcmread(patient_seg_dir + str('/1-1.dcm'))
            seg_reader = pydicom_seg.SegmentReader()
            seg_result = seg_reader.read(lung_seg_dcm)
            neoplasm_seg = np.flip(seg_result._segment_data[1], 0)
            # Find maximum tumour volume sice
            tumour_vols = np.sum(neoplasm_seg, axis=(1, 2))
            best_slice_ind = np.argmax(tumour_vols)
            range_slices = np.arange((best_slice_ind - (n_slices_scaled - 1) // 2),
                                     (best_slice_ind + (n_slices_scaled - 1) // 2 + 1))
            range_slices = range_slices[np.round(np.linspace(0, len(range_slices) - 1, n_slices)).astype(int)]
            if len(range_slices) == 0 or range_slices[0] >= patient_ct_pix.shape[0]:
                best_slices = patient_ct_pix_d[0:2]
            else:
                best_slices = patient_ct_pix_d[range_slices]
            radiogenomics_best_slices[i, 0, :, :] = np.mean(best_slices, axis=0)
            radiogenomics_best_slice_segmentations[i, 0, :, :] = (downscale_images(np.expand_dims(
                neoplasm_seg[np.argmax(tumour_vols)], 0), (dsize[0], dsize[1]))[0] > 0) * 1.
            if verbose > 0:
                bar.update(i)

        if not os.path.exists('../datasets/nsclc_lung/radiogenomics_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy')):
            if verbose > 0:
                print('Saving as a file...')
            np.save(file='../datasets/nsclc_lung/radiogenomics_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), arr=radiogenomics_best_slices, allow_pickle=True)
            np.save(file='../datasets/nsclc_lung/radiogenomics_segmentations_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), arr=radiogenomics_best_slice_segmentations,
                    allow_pickle=True)

    radiogenomics_best_slices = np.expand_dims(radiogenomics_best_slices, -1)
    return radiogenomics_best_slices


def load_radiogenomics_amc_images_max_tumour_volume_ave(radiogenomics_dir, n_slices, dsize, verbose=1):
    """
    Loads a subset of the NSCLC Radiogenomics dataset, takes an average 15 mm around the slice with the maximum
    transversal tumour area.
        https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics
    """
    assert n_slices % 2 == 1

    master_file = pd.read_csv(os.path.join(radiogenomics_dir, 'master_file_amc.csv'))

    radiogenomics_best_slices = np.zeros((len(master_file['Case ID'].values), 1, dsize[0], dsize[1]))

    if verbose > 0:
        print('Loading CT data:')
        print()

    if os.path.exists(
            '../datasets/nsclc_lung/radiogenomics_amc_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading from a pre-saved file...')
        radiogenomics_best_slices = np.load(
            file='../datasets/nsclc_lung/radiogenomics_amc_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=len(master_file['Case ID'].values))
            bar.start()

        for i in range(len(master_file['Case ID'].values)):
            patient_ct_dir = os.path.join(radiogenomics_dir, master_file['CT directory'].values[i])
            # Load CT
            results_dict = load_scan(patient_ct_dir)
            patient_ct_slices = results_dict['slices']
            slice_thickness = results_dict['slice_thickness']
            n_slices_scaled = int(REFERENCE_SLICE_THICKNESS / slice_thickness * n_slices)
            patient_ct_pix = get_pixels(patient_ct_slices)
            patient_ct_pix_d = downscale_images(patient_ct_pix, (dsize[0], dsize[1]))
            best_slice_ind = int(master_file['Tumor slice'].values[i])
            range_slices = np.arange((best_slice_ind - (n_slices_scaled - 1) // 2),
                                     (best_slice_ind + (n_slices_scaled - 1) // 2 + 1))
            range_slices = range_slices[np.round(np.linspace(0, len(range_slices) - 1, n_slices)).astype(int)]
            if len(range_slices) == 0 or range_slices[0] >= patient_ct_pix.shape[0]:
                best_slices = patient_ct_pix_d[0:2]
            else:
                best_slices = patient_ct_pix_d[range_slices]
            radiogenomics_best_slices[i, 0, :, :] = np.mean(best_slices, axis=0)
            if verbose > 0:
                bar.update(i)

        if not os.path.exists('../datasets/nsclc_lung/radiogenomics_amc_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy')):
            if verbose > 0:
                print('Saving as a file...')
            np.save(file='../datasets/nsclc_lung/radiogenomics_amc_best_slices_raw_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), arr=radiogenomics_best_slices, allow_pickle=True)

    radiogenomics_best_slices = np.expand_dims(radiogenomics_best_slices, -1)
    return radiogenomics_best_slices


def load_basel_images_max_tumour_volume_ave(basel_dir, n_slices, dsize, verbose=1):
    """
    Loads the dataset from the Basel University Hospital.
    Code adapted from Pattisapu et al.:
        https://github.com/pvk95/PAG
    """
    #
    assert n_slices % 2 == 1

    if verbose:
        print('Loading CT data:')
        print()

    if os.path.exists('../datasets/nsclc_lung/basel_best_slices_raw_' + str(n_slices) + '_' +
                      str(dsize[0]) + 'x' + str(dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading from a pre-saved file...')
        basel_best_slices = np.load(file='../datasets/nsclc_lung/basel_best_slices_raw_' + str(n_slices) +
                                         '_' + str(dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        scandates = pd.read_csv(os.path.join(os.path.dirname(basel_dir), 'TableS4.csv'))
        deathdates = pd.read_csv(os.path.join(os.path.dirname(basel_dir), 'deaths_lungstage_Basel.csv'))
        stages = pd.read_csv(os.path.join(os.path.dirname(basel_dir), 'lungstage-data/TNM_labels.csv'))
        metadata = pd.read_excel(os.path.join(os.path.dirname(basel_dir), 'lungstage-data/190418_Rohdaten_UICC_7.xlsx'),
                                 sheet_name=None, engine='openpyxl')
        metadata = metadata['SPSS total']

        nmissing_date = np.ones_like(deathdates['pat_death_date'].values).astype(bool)
        acc_codes = deathdates['accession'].values.astype('U32')

        scandates_first_saving = scandates['first saving'].values.astype('U32')
        scandates_id = scandates['ID'].values.astype('U32')
        for acode in acc_codes:
            if len(scandates_first_saving[scandates_id == acode]) == 0 or \
                    scandates_first_saving[scandates_id == acode] == 'nan':
                nmissing_date[acc_codes == acode] = False

        basel_n_patients = np.sum(nmissing_date)

        basel_best_slices = np.zeros((basel_n_patients, 1, dsize[0], dsize[1]))
        basel_segmentations = np.zeros((basel_n_patients, 1, dsize[0], dsize[1]))
        basel_tumor_volumes = np.zeros((basel_n_patients, 6))

        clinical_data = {'Scan date': np.repeat(pd.to_datetime('190821'), basel_n_patients),
                         'Death date': np.repeat(pd.to_datetime('190821'), basel_n_patients),
                         'Survival time': np.zeros((basel_n_patients,)),
                         'Event': np.ones((basel_n_patients,)), 'T': np.repeat('nan', basel_n_patients),
                         'N': np.repeat('nan', basel_n_patients), 'M': np.repeat('nan', basel_n_patients),
                         'Sex': np.zeros((basel_n_patients,)), 'Age': np.zeros((basel_n_patients,)),
                         'Real_lesions': np.zeros((basel_n_patients,)), 'UICC_I_IV': np.zeros((basel_n_patients,))}
        clinical_data = pd.DataFrame(data=clinical_data)

        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=basel_n_patients)
            bar.start()

        # Folders with images
        scan_dirs = []
        for d in DAYS:
            path = os.path.join(basel_dir, d)
            scan_dirs = scan_dirs + glob.glob(path + '*/')

        i = 0
        loaded_acc_codes = []
        for ct_dir in scan_dirs:
            # Patient's ID
            acc_code = ct_dir.split('ACC', 1)[1].replace('/', '')

            sav_date = pd.to_datetime(ct_dir.split('StagingD', 1)[1].split('T', 1)[0]).date()

            # Date of CT scan
            scan_date = scandates['first saving'].values.astype('U32')[scandates['ID'].values.astype(
                'U32') == acc_code]
            if len(scan_date) == 0 or scan_date[0] == 'nan':
                scan_date = None
            else:
                scan_date = pd.to_datetime(scan_date[0])

            # Date of death
            death_date = deathdates['pat_death_date'].values.astype('U32')[deathdates['accession'].values.astype(
                'U32') == acc_code]
            if len(death_date) == 0:
                death_date = None
            elif death_date[0] == 'nan' and scan_date is not None:
                death_date = pd.to_datetime('2021-08-23')
                clinical_data.at[i, 'Event'] = 0
            else:
                death_date = pd.to_datetime(death_date[0])

            sav_date_2 = scandates['saving for second reading'].values.astype('U32')[scandates['ID'].values.astype(
                'U32') == acc_code]
            if len(sav_date_2) == 0 or sav_date_2[0] == 'nan':
                sav_date_2 = None
            else:
                sav_date_2 = pd.to_datetime(sav_date_2[0]).date()

            # Only load CTs for patients with available survival data
            if scan_date is not None and death_date is not None and acc_code not in loaded_acc_codes:
                # Load the .npy file with the images.
                d = np.load(os.path.join(ct_dir, 'lsa.npz'))
                spacing = d['spacing']
                label_names = d['label_names']
                # Retrieve relevant slices of CT
                patient_ct_pix = d['CT'].astype(float)
                patient_ct_pix_d = downscale_images(patient_ct_pix, (dsize[0], dsize[1]))
                seg = d['Labels']
                tumour_vols = np.sum((seg > 0) * 1., axis=(1, 2))
                best_slice_ind = np.argmax(tumour_vols)
                best_slices = patient_ct_pix_d[(best_slice_ind -
                                                (n_slices - 1) // 2):(best_slice_ind + (n_slices - 1) // 2 + 1)]
                basel_best_slices[i, 0] = np.mean(best_slices, axis=0)
                best_slice_seg = (seg[best_slice_ind] > 0) * 1.
                basel_segmentations[i, 0] = downscale_images(np.expand_dims(best_slice_seg, 0), (dsize[0], dsize[1]))[0]
                basel_tumor_volumes[i, 0] = np.sum(tumour_vols[(best_slice_ind -
                                                                (n_slices - 1) // 2):(
                                                                           best_slice_ind + (n_slices - 1) // 2 + 1)])
                basel_tumor_volumes[i, 1] = np.sum(tumour_vols[(best_slice_ind -
                                                                (n_slices - 1) // 2):(
                                                                           best_slice_ind + (n_slices - 1) // 2 + 1)] *
                                                   spacing[0] * spacing[1] * spacing[2])
                basel_tumor_volumes[i, 3] = np.sum(tumour_vols)
                basel_tumor_volumes[i, 4] = np.sum(tumour_vols) * spacing[0] * spacing[1] * spacing[2]

                # Find relevant metadata
                sex = metadata['sex'].values[metadata['id'].values == int(acc_code)]
                if len(sex) == 0:
                    sex = 'nan'
                    age = 'nan'
                    uicc = 'nan'
                    lesions = 'nan'
                else:
                    sex = sex[0]
                    age = metadata['age'].values[metadata['id'].values == int(acc_code)][0]
                    uicc = metadata['UICC_I_IV'].values[metadata['id'].values == int(acc_code)][0]
                    lesions = metadata['real_lesions'].values[metadata['id'].values == int(acc_code)][0]
                T = stages['T'].values[stages['Accession#'] == int(acc_code)]
                if len(T) == 0:
                    T = 'nan'
                    M = 'nan'
                    N = 'nan'
                else:
                    T = T[0]
                    M = stages['M'].values[stages['Accession#'] == int(acc_code)][0]
                    N = stages['N'].values[stages['Accession#'] == int(acc_code)][0]

                # Save clinical data
                clinical_data.at[i, 'Scan date'] = scan_date
                clinical_data.at[i, 'Death date'] = death_date
                clinical_data.at[i, 'Survival time'] = (death_date - scan_date).days
                clinical_data.at[i, 'Sex'] = sex
                clinical_data.at[i, 'Age'] = age
                clinical_data.at[i, 'UICC_I_IV'] = uicc
                clinical_data.at[i, 'Real_lesions'] = lesions
                clinical_data.at[i, 'T'] = T
                clinical_data.at[i, 'M'] = M
                clinical_data.at[i, 'N'] = N

                loaded_acc_codes.append(acc_code)
                if verbose:
                    bar.update(i)
                i = i + 1

        basel_best_slices = basel_best_slices[0:i]
        basel_segmentations = basel_segmentations[0:i]
        basel_tumor_volumes = basel_tumor_volumes[0:i]
        clinical_data = clinical_data[0:i]

        if not os.path.exists('../datasets/nsclc_lung/basel_best_slices_raw_' + str(n_slices) +
                              '_' + str(dsize[0]) + 'x' + str(dsize[1]) + str('.npy')):
            if verbose:
                print('Saving as a file...')
            np.save(file='../datasets/nsclc_lung/basel_best_slices_raw_' + str(n_slices) + '_' + str(dsize[0]) +
                         'x' + str(dsize[1]) + str('.npy'), arr=basel_best_slices, allow_pickle=True)
            # Save segmentations
            np.save(file='../datasets/nsclc_lung/basel_segmentations_' + str(n_slices) + '_' + str(dsize[0]) +
                         'x' + str(dsize[1]) + str('.npy'), arr=basel_segmentations, allow_pickle=True)
            # Save segmentations
            np.save(file='../datasets/nsclc_lung/basel_tumor_volumes.npy', arr=basel_tumor_volumes, allow_pickle=True)
            # Save clinical data
            clinical_data.to_csv('../datasets/nsclc_lung/clinical_data_basel.csv', index=False)

    basel_best_slices = np.expand_dims(basel_best_slices, -1)
    return basel_best_slices


def preprocess_lung1_images(lung1_dir, n_slices, dsize, n_bins=40, verbose=1):
    """
    Preprocesses Lung1 CT images.
    """
    if os.path.exists(
            '../datasets/nsclc_lung/lung1_best_slices_preprocessed_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                    dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading preprocessed data from a pre-saved file...')
        X = np.load(file='../datasets/nsclc_lung/lung1_best_slices_preprocessed_' + str(n_slices) + '_' + str(
            dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        # load data
        X = load_lung1_images_max_tumour_volume_ave(lung1_dir, n_slices, dsize)
        X = np.reshape(X, (-1, X.shape[1], dsize[0], dsize[1]))
        # preprocess
        if verbose > 0:
            print('Preprocess data...')
        bar = progressbar.ProgressBar(maxval=len(X))
        bar.start()

        of_30 = [21, 147]
        p_10 = [50, 251, 304]
        for i in range(len(X)):
            offset = 50
            p_min = 0.15
            if i in of_30:
                offset = 30
            if i in p_10:
                p_min = 0.10
            elif i == 314:
                p_min = 0.2
            temp = np.copy(X[i])
            temp = normalise_images(temp)
            if i == 28:
                temp = temp[:, offset + 45:(X.shape[2] - offset - 25), offset + 20:(X.shape[3] - offset - 5)]
            elif i == 303:
                temp = temp[:, offset + 20:(X.shape[2] - offset - 30), offset + 20:(X.shape[3] - offset - 5)]
            elif i in [10, 36, 106, 292, 354]:
                temp = temp[:, offset:(X.shape[2] - offset), offset + 20:(X.shape[3] - offset + 10)]
            elif i == 351:
                temp = temp[:, 40 + offset:(X.shape[2] - offset - 10), offset + 30:(X.shape[3] - offset)]
            elif i in [9, 46, 57, 67, 78, 132, 142, 146, 257, 302]:
                temp = temp[:, offset:(X.shape[2] - offset), offset:(X.shape[3] - offset + 30)]
            elif i in [4, 26, 51, 129, 159, 199, 292, 137]:
                temp = temp[:, offset:(X.shape[2] - offset), offset - 30:(X.shape[3] - offset)]
            elif i in [168, 281]:
                temp = temp[:, -20 + offset:(X.shape[2] - offset), offset - 30:(X.shape[3] - offset) + 20]
            else:
                temp = temp[:, offset:(X.shape[2] - offset), offset:(X.shape[3] - offset)]
            X[i] = crop_equalize_images(temp, dsize[0], n_bins=n_bins, lung_segmentations=None,
                                        p_min=p_min)
            bar.update(i)
        X = np.delete(X, IGNORED_PATIENTS, axis=0)
        if verbose > 0:
            print('Saving as a file...')
        np.save(
            file='../datasets/nsclc_lung/lung1_best_slices_preprocessed_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), arr=X, allow_pickle=True)
    X = np.expand_dims(X, -1)
    return X


def preprocess_lung3_images(lung3_dir, n_slices, dsize, n_bins=40, verbose=1):
    """
    Preprocesses Lung3 CT images.
    """
    if os.path.exists(
            '../datasets/nsclc_lung/lung3_best_slices_preprocessed_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                    dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading preprocessed data from a pre-saved file...')
        X = np.load(file='../datasets/nsclc_lung/lung3_best_slices_preprocessed_' + str(n_slices) + '_' + str(
            dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        # load data
        X = load_lung3_images_max_tumour_volume_ave(lung3_dir, n_slices, dsize)
        X = np.reshape(X, (-1, X.shape[1], dsize[0], dsize[1]))
        # preprocess
        if verbose > 0:
            print('Preprocess data...')
        bar = progressbar.ProgressBar(maxval=len(X))
        bar.start()

        of_20 = [1, 6, 31, 33, 62]
        of_0 = [21, 50, 54, 60, 69, 10]
        for i in range(len(X)):
            offset = 30
            if i in of_20:
                offset = 20
            elif i in of_0:
                offset = 0

            temp = np.copy(X[i])
            temp = normalise_images(temp)
            if i == 10:
                temp = temp[:, offset + 70:(X.shape[2] - offset), offset:(X.shape[3] - offset)]
            elif i == 21:
                temp = temp[:, offset + 110:(X.shape[2] - offset), offset:(X.shape[3] - offset)]
            else:
                temp = temp[:, offset:(X.shape[2] - offset), offset:(X.shape[3] - offset)]

            X[i] = crop_equalize_images(temp, dsize[0], n_bins=n_bins, lung_segmentations=None)
            bar.update(i)
        if verbose > 0:
            print('Saving as a file...')
        np.save(
            file='../datasets/nsclc_lung/lung3_best_slices_preprocessed_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), arr=X, allow_pickle=True)
    X = np.delete(X, IGNORED_LUNG3_PATIENTS, axis=0)
    X = np.expand_dims(X, -1)
    return X


def preprocess_radiogenomics_images(radiogenomics_dir, n_slices, dsize, n_bins=40, verbose=1):
    """
    Preprocesses a subset of NSCLC Radiogenomics CT images.
    """
    if os.path.exists('../datasets/nsclc_lung/radiogenomics_best_slices_preprocessed_' + str(n_slices) + '_' + str(
            dsize[0]) + 'x' + str(dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading preprocessed data from a pre-saved file...')
        X = np.load(file='../datasets/nsclc_lung/radiogenomics_best_slices_preprocessed_' + str(n_slices) + '_' + str(
            dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        # load data
        X = load_radiogenomics_images_max_tumour_volume_ave(radiogenomics_dir, n_slices, dsize)
        X = np.reshape(X, (-1, X.shape[1], dsize[0], dsize[1]))
        lung_segmentations = None
        if verbose > 0:
            print('Preprocess data...')
        bar = progressbar.ProgressBar(maxval=len(X))
        bar.start()

        of_30 = [6, 14, 16, 26, 29, 71, 81, 90, 95]
        of_15 = [63, 77]
        of_25 = [91, 92, 35, 70]
        p_10 = [6, 14, 71]
        for i in range(len(X)):
            offset = 35
            p_min = 0.15
            if i in of_30:
                offset = 30
            elif i in of_25:
                offset = 25
            elif i in of_15:
                offset = 15
            if i in p_10:
                p_min = 0.10
            temp = np.copy(X[i])
            if np.sum(temp <= 0) >= 300:
                # Remove the circle pattern
                if i in [6, 14, 71, 63]:
                    temp = temp[:, offset + 35:(temp.shape[0] - offset - 20), offset:(temp.shape[1] - offset)]
                elif i in [81]:
                    temp = temp[:, offset + 10:(temp.shape[0] - offset - 40), offset:(temp.shape[1] - offset)]
                elif i in [77, 91, 92]:
                    temp = temp[:, offset + 30:(temp.shape[0] - offset - 40), offset:(temp.shape[1] - offset)]
                elif i in [16, 29, 95]:
                    temp = temp[:, offset + 15:(temp.shape[0] - offset - 20), offset:(temp.shape[1] - offset)]
                elif i in [26, 90]:
                    temp = temp[:, offset + 20:(temp.shape[0] - offset - 20), offset - 5:(temp.shape[1] - offset - 10)]
                elif i in [35]:
                    temp = temp[:, offset:(temp.shape[0] - offset - 30), offset:(temp.shape[1] - offset)]
                elif i in [70]:
                    temp = temp[:, offset + 35:(temp.shape[0] - offset - 20), offset - 10:(temp.shape[1] - offset)]
                else:
                    temp = temp[:, offset + 10:(temp.shape[0] - offset - 10), offset:(temp.shape[1] - offset)]

            temp = normalise_images(temp)
            X[i] = crop_equalize_images(temp, dsize[0], n_bins=n_bins, lung_segmentations=lung_segmentations)
            bar.update(i)
        if verbose > 0:
            print('Saving as a file...')

        X = np.delete(X, IGNORED_RADIOGENOMICS_PATIENTS, axis=0)
        np.save(
            file='../datasets/nsclc_lung/radiogenomics_best_slices_preprocessed_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), arr=X, allow_pickle=True)
    X = np.expand_dims(X, -1)
    return X


def preprocess_radiogenomics_images_amc(radiogenomics_dir, n_slices, dsize, n_bins=40, verbose=1):
    """
    Preprocesses a subset of NSCLC Radiogenomics CT images.
    """
    if os.path.exists('../datasets/nsclc_lung/radiogenomics_amc_best_slices_preprocessed_' + str(n_slices) + '_' + str(
            dsize[0]) + 'x' + str(dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading preprocessed data from a pre-saved file...')
        X = np.load(
            file='../datasets/nsclc_lung/radiogenomics_amc_best_slices_preprocessed_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        # load data
        X = load_radiogenomics_amc_images_max_tumour_volume_ave(radiogenomics_dir, n_slices, dsize)
        X = np.reshape(X, (-1, X.shape[1], dsize[0], dsize[1]))
        lung_segmentations = None
        if verbose > 0:
            print('Preprocess data...')
        bar = progressbar.ProgressBar(maxval=len(X))
        bar.start()

        of_25 = [3, 8, 10, 15, 16, 17, 28, 39, 38, 36]
        of_40 = [29, 40]
        for i in range(len(X)):
            offset = 45
            if i in of_40:
                offset = 40
            elif i in of_25:
                offset = 25
            temp = np.copy(X[i])
            if np.sum(temp <= 0) >= 300:
                # Remove the circle pattern
                if i == 0:
                    temp = temp[:, offset:(temp.shape[0] - offset), offset - 10:(temp.shape[1] - offset)]
                elif i == 29:
                    temp = temp[:, offset + 35:(temp.shape[0] - offset), offset:(temp.shape[1] - offset + 20)]
                else:
                    temp = temp[:, offset:(temp.shape[0] - offset), offset:(temp.shape[1] - offset)]
            temp = normalise_images(temp)
            X[i] = crop_equalize_images(temp, dsize[0], n_bins=n_bins, lung_segmentations=lung_segmentations)
            bar.update(i)
        if verbose > 0:
            print('Saving as a file...')
        np.save(
            file='../datasets/nsclc_lung/radiogenomics_amc_best_slices_preprocessed_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), arr=X, allow_pickle=True)
    X = np.expand_dims(X, -1)
    return X


def preprocess_basel_images(basel_dir, n_slices, dsize, n_bins=40, verbose=1):
    """
    Preprocesses Basel University Hospital CT images.
    """
    if os.path.exists(
            '../datasets/nsclc_lung/basel_best_slices_preprocessed_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + str(
                    dsize[1]) + str('.npy')):
        if verbose > 0:
            print('Loading preprocessed data from a pre-saved file...')
        X = np.load(file='../datasets/nsclc_lung/basel_best_slices_preprocessed_' + str(n_slices) + '_' + str(
            dsize[0]) + 'x' + str(dsize[1]) + str('.npy'), allow_pickle=True)
    else:
        # load data
        X = load_basel_images_max_tumour_volume_ave(basel_dir, n_slices, dsize)
        X = np.reshape(X, (-1, X.shape[1], dsize[0], dsize[1]))
        lung_segmentations = None
        if verbose:
            print('Preprocess data...')
        bar = progressbar.ProgressBar(maxval=len(X))
        bar.start()

        of_25 = [36, 72, 80, 85, 104, 108, 137, 344, 351]
        of_60 = [125, 202, 203, 357, 360, 320]
        of_30 = [200]
        p_06 = [125]
        p_09 = [301]
        p_14 = [71]
        p_78 = [200]
        plus_20 = [202, 203, 357, 360]
        plus_10 = [320]
        minus_20 = [186]
        for i in range(len(X)):
            p_min = 0.15
            offset = 45
            if i in of_25:
                offset = 25
            elif i in of_60:
                offset = 60
            elif i in of_30:
                offset = 30
            if i in p_06:
                p_min = 0.06
            elif i in p_09:
                p_min = 0.09
            elif i in p_14:
                p_min = 0.14
            elif i in p_78:
                p_min = 0.78
            temp = np.copy(X[i])
            if np.sum(temp <= 0) >= 300:
                # Remove the circle pattern
                if i in plus_20:
                    temp = temp[:, offset:(temp.shape[0] - offset), offset:(temp.shape[1] - offset + 20)]
                elif i in plus_10:
                    temp = temp[:, offset:(temp.shape[0] - offset), offset:(temp.shape[1] - offset + 10)]
                elif i in minus_20:
                    temp = temp[:, offset:(temp.shape[0] - offset), offset:(temp.shape[1] - offset - 20)]
                else:
                    temp = temp[:, offset:(temp.shape[0] - offset), offset:(temp.shape[1] - offset)]
            temp = normalise_images(temp)
            X[i] = crop_equalize_images(temp, dsize[0], n_bins=n_bins, lung_segmentations=lung_segmentations,
                                        p_min=p_min)
            bar.update(i)
        if verbose:
            print('Saving as a file...')
        X = np.delete(X, IGNORED_BASEL_PATIENTS, axis=0)
        np.save(
            file='../datasets/nsclc_lung/basel_best_slices_preprocessed_' + str(n_slices) + '_' + str(
                dsize[0]) + 'x' + str(
                dsize[1]) + str('.npy'), arr=X, allow_pickle=True)

    X = np.expand_dims(X, -1)
    return X


def augment_images(images):
    """
    Augments a batch of CT images.
    """
    images = np.squeeze(images)
    images_augmented = np.zeros(images.shape)
    for i in range(images.shape[0]):
        image = np.squeeze(images[i])

        image = augment_brightness(image, value_min=-0.1, value_max=0.1)

        o = np.random.rand()
        if o < 0.5:
            image = augment_noise(image)

        o = np.random.rand()
        if o < 0.5:
            image = np.flip(image, axis=1)

        o = np.random.rand()
        if o < 0.5:
            image = augment_rotate(image, angle_min=-4, angle_max=4)

        o = np.random.rand()
        if o < 0.5:
            image = augment_blur(image, width_min=1, width_max=3)

        image = augment_zoom(image, ratio_min=0.9, ratio_max=1.1)
        o = np.random.rand()
        if o > 0.5:
            image = augment_stretch_horizontal(image, ratio_min=1.0, ratio_max=1.2)
        else:
            image = augment_stretch_vertical(image, ratio_min=1.0, ratio_max=1.1)

        image = augment_shift(image, shift_h_min=-0.1, shift_h_max=0.1, shift_v_min=-0.1, shift_v_max=0.1)
        images_augmented[i] = np.squeeze(image)

    images_augmented = np.expand_dims(images_augmented, -1)
    return images_augmented


# Atomic augmentations for CT scans
def augment_rotate(image, angle_min=-30, angle_max=30):
    theta = np.random.uniform(angle_min, angle_max)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def augment_blur(image, width_min=2, width_max=10):
    w = np.random.randint(width_min, width_max + 1)
    blurred = cv2.blur(image, (w, w))

    return blurred


def augment_sharpen(image):
    sh_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image_sharpened = cv2.filter2D(image, -1, sh_filter)
    image_sharpened = image_sharpened - np.min(image_sharpened)
    image_sharpened = image_sharpened / np.max(image_sharpened)

    return image_sharpened


def augment_noise(image):
    noise_mask = np.random.poisson(np.abs(image) * 255 / np.max(image))
    image_noisy = image + noise_mask

    image_noisy = cv2.normalize(image_noisy, dst=None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    return image_noisy


def augment_invert(image):
    image_inverted = 1. - image

    return image_inverted


def augment_zoom(image, ratio_min=0.8, ratio_max=1.2, pad_zeros=True):
    ratio = np.random.uniform(ratio_min, ratio_max)
    image_rescaled = cv2.resize(image, dsize=(int(image.shape[0] * ratio), int(image.shape[1] * ratio)),
                                interpolation=cv2.INTER_CUBIC)
    outer_brim = np.concatenate((image[:, -1], image[-1, :], image[0, :], image[:, 0]))
    if pad_zeros:
        image_zoomed = np.zeros_like(image)
    else:

        image_zoomed = np.ones_like(image) * np.median(outer_brim)
    if ratio < 1.0:
        # Pad
        ht = image_rescaled.shape[0]
        wd = image_rescaled.shape[1]
        # Compute center offset
        xx = (image.shape[0] - wd) // 2
        yy = (image.shape[1] - ht) // 2
        image_zoomed[yy:yy + ht, xx:xx + wd] = image_rescaled
    else:
        # Crop
        center = [image_rescaled.shape[0] / 2, image_rescaled.shape[1] / 2]
        x = center[1] - image.shape[1] / 2
        y = center[0] - image.shape[0] / 2
        image_zoomed = image_rescaled[int(y):int(y + image.shape[0]), int(x):int(x + image.shape[1])]
    return image_zoomed


def augment_shift(image, shift_h_min=-0.1, shift_h_max=0.1, shift_v_min=-0.1, shift_v_max=0.1, pad_zeros=True):
    shift_vertical = np.random.uniform(shift_v_min, shift_v_max)
    shift_horizontal = np.random.uniform(shift_h_min, shift_h_max)
    outer_brim = np.concatenate((image[:, -1], image[-1, :], image[0, :], image[:, 0]))
    if pad_zeros:
        image_shifted = np.zeros_like(image)
    else:
        image_shifted = np.ones_like(image) * np.median(outer_brim)
    if shift_vertical < 0:
        x0 = int(-shift_vertical * image.shape[0])
        x1 = image.shape[0] - 1
        x0_dest = 0
        x1_dest = int(image.shape[0] + shift_vertical * image.shape[0])
    else:
        x0 = 0
        x1 = int(image.shape[0] - shift_vertical * image.shape[0])
        x0_dest = int(shift_vertical * image.shape[0])
        x1_dest = image.shape[0] - 1
    if shift_horizontal < 0:
        y0 = int(-shift_horizontal * image.shape[1])
        y1 = image.shape[1] - 1
        y0_dest = 0
        y1_dest = int(image.shape[1] + shift_horizontal * image.shape[1])
    else:
        y0 = 0
        y1 = int(image.shape[1] - shift_horizontal * image.shape[1])
        y0_dest = int(shift_horizontal * image.shape[1])
        y1_dest = image.shape[1] - 1
    image_shifted[x0_dest:x1_dest, y0_dest:y1_dest] = image[x0:x1, y0:y1]
    return image_shifted


def augment_stretch_horizontal(image, ratio_min=0.8, ratio_max=1.2, pad_zeros=True):
    ratio = np.random.uniform(ratio_min, ratio_max)
    outer_brim = np.concatenate((image[:, -1], image[-1, :], image[0, :], image[:, 0]))
    if pad_zeros:
        image_stretched = np.zeros_like(image)
    else:
        image_stretched = np.ones_like(image) * np.median(outer_brim)
    image_rescaled = cv2.resize(image, dsize=(int(image.shape[0] * ratio), image.shape[1]),
                                interpolation=cv2.INTER_CUBIC)
    if ratio < 1.0:
        # Pad
        ht = image_rescaled.shape[0]
        wd = image_rescaled.shape[1]
        # Compute center offset
        xx = (image.shape[0] - wd) // 2
        yy = (image.shape[1] - ht) // 2
        image_stretched[:, xx:xx + wd] = image_rescaled
    else:
        # Crop
        center = [image_rescaled.shape[0] / 2, image_rescaled.shape[1] / 2]
        x = center[1] - image.shape[1] / 2
        y = center[0] - image.shape[0] / 2
        image_stretched = image_rescaled[:, int(x):int(x + image.shape[1])]
    return image_stretched


def augment_stretch_vertical(image, ratio_min=0.8, ratio_max=1.2, pad_zeros=True):
    ratio = np.random.uniform(ratio_min, ratio_max)
    outer_brim = np.concatenate((image[:, -1], image[-1, :], image[0, :], image[:, 0]))
    if pad_zeros:
        image_stretched = np.zeros_like(image)
    else:
        image_stretched = np.ones_like(image) * np.median(outer_brim)
    image_rescaled = cv2.resize(image, dsize=(image.shape[0], int(image.shape[1] * ratio)),
                                interpolation=cv2.INTER_CUBIC)
    if ratio < 1.0:
        # Pad
        ht = image_rescaled.shape[0]
        wd = image_rescaled.shape[1]
        # Compute center offset
        xx = (image.shape[0] - wd) // 2
        yy = (image.shape[1] - ht) // 2
        image_stretched[yy:yy + ht, :] = image_rescaled
    else:
        # Crop
        center = [image_rescaled.shape[0] / 2, image_rescaled.shape[1] / 2]
        x = center[1] - image.shape[1] / 2
        y = center[0] - image.shape[0] / 2
        image_stretched = image_rescaled[int(y):int(y + image.shape[0]), :]
    return image_stretched


def augment_brightness(image, value_min=-0.1, value_max=0.1):
    u = (np.random.uniform(0, 1) >= 0.5) * 1.0
    value = u * np.random.uniform(value_min, value_min / 2.0) + (1 - u) * np.random.uniform(value_max / 2.0, value_max)
    if value >= 0:
        image_augmented = np.where((1.0 - image) < value, 1.0, image + value)
    else:
        image_augmented = np.where(image < value, 0.0, image + value)
    return image_augmented
