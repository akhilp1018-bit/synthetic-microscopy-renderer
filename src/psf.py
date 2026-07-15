"""
Point spread function utilities.

This module handles:
    - loading PSF TIFF files
    - converting PSF arrays to ZYX order
    - optionally converting a 1-photon PSF to a simple 2-photon-like PSF
    - generating an analytical Gaussian PSF

Coordinate convention:
    PSF arrays are returned in ZYX order:
    [Z slices, Y pixels, X pixels].
"""

import numpy as np
import tifffile


def _move_psf_to_zyx(arr: np.ndarray) -> np.ndarray:
    """
    Move a 3D PSF array to ZYX order.

    Some PSF files may not be saved with Z as the first axis. This function
    assumes the Z axis is the smallest dimension and moves it to axis 0.

    Args:
        arr:
            Input 3D PSF array.

    Returns:
        PSF array in ZYX order.
    """
    if arr.ndim != 3:
        raise ValueError(f"PSF must be 3D, got shape {arr.shape}")

    z_axis = int(np.argmin(arr.shape))

    if z_axis != 0:
        arr = np.moveaxis(arr, z_axis, 0)

    return arr


def load_psf_zyx(
    path: str,
    two_photon_like: bool = False,
    clip_negative: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load a PSF TIFF file and return a normalized ZYX PSF.

    Args:
        path:
            Path to PSF TIFF file.
        two_photon_like:
            If True, square the PSF before normalization. This is a simple
            approximation for a two-photon-like excitation profile.
        clip_negative:
            If True, negative values are clipped to zero.
        verbose:
            If True, print PSF information.

    Returns:
        Normalized PSF as float32 NumPy array in ZYX order.
    """
    arr = tifffile.imread(path).astype(np.float32)
    arr = _move_psf_to_zyx(arr)

    if clip_negative:
        arr = np.maximum(arr, 0.0)

    if two_photon_like:
        arr = arr ** 2

    arr /= arr.sum() + 1e-12

    if verbose:
        print("Loaded PSF:")
        print(f"  path            = {path}")
        print(f"  shape ZYX       = {arr.shape}")
        print(f"  two_photon_like = {two_photon_like}")
        print(f"  sum             = {arr.sum():.6f}")
        print(f"  max             = {arr.max():.6e}")

    return arr.astype(np.float32)


def fwhm_to_sigma(fwhm: float) -> float:
    """
    Convert full width at half maximum to Gaussian sigma.
    """
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def make_gaussian_psf_matched_zyx(
    shape_zyx=(13, 65, 65),
    lambda_nm=488.0,
    na=1.0,
    n=1.33,
    xy_um_per_px=0.094,
    z_step_um=0.5,
    sigma_scale_xy=1.0,
    sigma_scale_z=1.0,
    two_photon_like=True,
    verbose=True,
) -> np.ndarray:
    """
    Create an analytical Gaussian PSF matched to the image sampling.

    The Gaussian width is estimated from simple diffraction-based FWHM
    approximations and converted into pixel units using the configured XY and Z
    sampling.

    Args:
        shape_zyx:
            Output PSF shape in [Z, Y, X] order.
        lambda_nm:
            Excitation/emission wavelength parameter in nanometres.
        na:
            Numerical aperture.
        n:
            Refractive index.
        xy_um_per_px:
            XY pixel size in micrometres.
        z_step_um:
            Z slice spacing in micrometres.
        sigma_scale_xy:
            Optional scale factor for lateral PSF width.
        sigma_scale_z:
            Optional scale factor for axial PSF width.
        two_photon_like:
            If True, square the Gaussian PSF before normalization.
        verbose:
            If True, print PSF information.

    Returns:
        Normalized Gaussian PSF as float32 NumPy array in ZYX order.
    """
    pz, py, px = map(int, shape_zyx)

    lam_um = lambda_nm * 1e-3

    # Simple diffraction-based FWHM approximations.
    fwhm_xy_um = 0.61 * lam_um / na
    fwhm_z_um = (2.0 * n * lam_um) / (na ** 2)

    sigma_xy_um = fwhm_to_sigma(fwhm_xy_um)
    sigma_z_um = fwhm_to_sigma(fwhm_z_um)

    sigma_x_px = (sigma_xy_um / xy_um_per_px) * sigma_scale_xy
    sigma_y_px = (sigma_xy_um / xy_um_per_px) * sigma_scale_xy
    sigma_z_px = (sigma_z_um / z_step_um) * sigma_scale_z

    if verbose:
        print("Gaussian PSF matched:")
        print(f"  shape ZYX       = {shape_zyx}")
        print(f"  lambda_nm       = {lambda_nm}")
        print(f"  NA              = {na}")
        print(f"  n               = {n}")
        print(f"  xy_um_per_px    = {xy_um_per_px}")
        print(f"  z_step_um       = {z_step_um}")
        print(f"  two_photon_like = {two_photon_like}")

    z = np.arange(pz, dtype=np.float32) - (pz // 2)
    y = np.arange(py, dtype=np.float32) - (py // 2)
    x = np.arange(px, dtype=np.float32) - (px // 2)

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    psf = np.exp(
        -(
            zz**2 / (2.0 * sigma_z_px**2)
            + yy**2 / (2.0 * sigma_y_px**2)
            + xx**2 / (2.0 * sigma_x_px**2)
        )
    ).astype(np.float32)

    if two_photon_like:
        psf = psf ** 2

    psf /= psf.sum() + 1e-12

    return psf.astype(np.float32)