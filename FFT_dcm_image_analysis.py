import numpy as np
import pydicom
import argparse
from scipy.fftpack import fft2, fftshift

def compute_match_score(dicom_file, sagittal_slice_idx, periodicity):
    """
    Detects periodically appearing horizontal lines in a specified sagittal slice of a DICOM image and computes a match score.

    Parameters:
        dicom_file (str): Path to the DICOM file.
        sagittal_slice_idx (int): Index of the sagittal slice to analyze (starting from 0).
        periodicity (int): Periodicity of the horizontal lines in pixels.

    Returns:
        float: Match score (energy of target frequency / total energy).
    """
    # Load the DICOM file
    dicom_data = pydicom.dcmread(dicom_file)

    # Get the image data (assumed to be a 3D image, e.g., a CT scan or MRI)
    image_data = dicom_data.pixel_array  # Assuming this is a NumPy array

    # Check if the image is 3D
    if len(image_data.shape) != 3:
        raise ValueError("DICOM file does not contain a 3D volume.")

    # Extract the sagittal slice (typically the 3rd dimension)
    # Assuming the sagittal slice corresponds to the 3rd dimension in a 3D image
    image = image_data[:, :, sagittal_slice_idx]  # Select the sagittal slice

    # Flip the image by 180 degrees (vertically)
    flipped_image = np.flipud(image)

    # Perform Fourier Transform
    F = fft2(flipped_image)
    F_shifted = fftshift(F)
    magnitude_spectrum = np.abs(F_shifted)

    # Identify the target frequency index
    freq_index = int(flipped_image.shape[0] / periodicity)

    # Create a mask to isolate the target frequency band
    bandwidth = 5  # Allow a small range around the target frequency
    mask = np.zeros_like(F_shifted, dtype=bool)
    mask[freq_index - bandwidth:freq_index + bandwidth, :] = True

    # Compute the total energy and energy in the target frequency band
    total_energy = np.sum(magnitude_spectrum)
    target_energy = np.sum(magnitude_spectrum[mask])

    # Compute the match score
    match_score = target_energy / total_energy
    return match_score

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Detect periodic horizontal lines in a flipped sagittal slice of a DICOM image.")
    parser.add_argument("dicom_file", type=str, help="Path to the DICOM file (with its extension)")
    parser.add_argument("sagittal_slice_idx", type=int, help="Index of the sagittal slice to analyze (starting from 0)")
    parser.add_argument("periodicity", type=int, help="Periodicity of the horizontal lines in pixels")
    args = parser.parse_args()

    # Compute the match score
    try:
        match_score = compute_match_score(args.dicom_file, args.sagittal_slice_idx, args.periodicity)
        # Print only the match score number (no label)
        print(f"{match_score:.4f}")
    except Exception as e:
        print(f"Error: {e}")
