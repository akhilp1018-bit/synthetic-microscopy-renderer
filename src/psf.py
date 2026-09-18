"""
Point spread function utilities.

This module handles:
    - loading Born-Wolf PSF TIFF files
    - converting PSF arrays to ZYX order
    - generating an analytical Gaussian two-photon excitation PSF

Coordinate convention:
    PSF arrays are returned in ZYX order:
    [Z slices, Y pixels, X pixels].
"""

import numpy as np
import tifffile


def _move_psf_to_zyx(arr: np.ndarray) -> np.ndarray:
    """Move a 3D PSF array to ZYX order."""
    if arr.ndim != 3:
        raise ValueError(f"PSF must be 3D, got shape {arr.shape}")

    z_axis = int(np.argmin(arr.shape))

    if z_axis != 0:
        arr = np.moveaxis(arr, z_axis, 0)

    return arr


def load_psf_zyx(
    path: str,
    clip_negative: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load a Born-Wolf PSF TIFF file and return a normalized ZYX PSF.

    The loaded PSF is used directly after optional negative-value clipping
    and normalization.
    """
    arr = tifffile.imread(path).astype(np.float32)
    arr = _move_psf_to_zyx(arr)

    if clip_negative:
        arr = np.maximum(arr, 0.0)

    arr /= arr.sum() + 1e-12

    if verbose:
        print("Loaded Born-Wolf PSF:")
        print(f"  path      = {path}")
        print(f"  shape ZYX = {arr.shape}")
        print(f"  sum       = {arr.sum():.6f}")
        print(f"  max       = {arr.max():.6e}")

    return arr.astype(np.float32)


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert full width at half maximum to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def make_gaussian_psf_matched_zyx(
    shape_zyx=(13, 65, 65),
    excitation_lambda_nm=920.0,
    na=1.0,
    n=1.33,
    xy_um_per_px=0.094,
    z_step_um=0.5,
    sigma_scale_xy=1.0,
    sigma_scale_z=1.0,
    verbose=True,
) -> np.ndarray:
    """
    Create an analytical Gaussian approximation of a two-photon excitation PSF.

    The effective lateral and axial FWHM are estimated directly from the
    two-photon excitation wavelength. The resulting Gaussian is not squared
    again after construction.

    The approximations used are:

        lateral FWHM:
            0.541 * lambda / (sqrt(2) * NA**0.91)

        axial FWHM:
            (0.886 * lambda / sqrt(2))
            / (n - sqrt(n**2 - NA**2))

    Args:
        shape_zyx:
            Output PSF shape in [Z, Y, X] order.
        excitation_lambda_nm:
            Two-photon excitation wavelength in nanometres.
        na:
            Numerical aperture.
        n:
            Refractive index.
        xy_um_per_px:
            XY pixel size in micrometres.
        z_step_um:
            Z slice spacing in micrometres.
        sigma_scale_xy:
            Optional scale factor for lateral Gaussian sigma.
        sigma_scale_z:
            Optional scale factor for axial Gaussian sigma.
        verbose:
            If True, print PSF parameters.

    Returns:
        Normalized Gaussian PSF as float32 NumPy array in ZYX order.
    """
    pz, py, px = map(int, shape_zyx)

    if excitation_lambda_nm <= 0:
        raise ValueError("excitation_lambda_nm must be positive.")
    if na <= 0:
        raise ValueError("na must be positive.")
    if n <= 0:
        raise ValueError("refractive index n must be positive.")
    if na >= n:
        raise ValueError(
            "For this axial PSF approximation, NA must be smaller than "
            "the refractive index n."
        )

    lam_um = float(excitation_lambda_nm) * 1e-3

    fwhm_xy_um = (
        0.541 * lam_um
        / (np.sqrt(2.0) * (na ** 0.91))
    )

    fwhm_z_um = (
        (0.886 * lam_um / np.sqrt(2.0))
        / (n - np.sqrt(n**2 - na**2))
    )

    sigma_xy_um = fwhm_to_sigma(fwhm_xy_um)
    sigma_z_um = fwhm_to_sigma(fwhm_z_um)

    sigma_x_px = (sigma_xy_um / xy_um_per_px) * sigma_scale_xy
    sigma_y_px = (sigma_xy_um / xy_um_per_px) * sigma_scale_xy
    sigma_z_px = (sigma_z_um / z_step_um) * sigma_scale_z

    if verbose:
        print("Gaussian two-photon PSF:")
        print(f"  shape ZYX             = {shape_zyx}")
        print(f"  excitation_lambda_nm  = {excitation_lambda_nm}")
        print(f"  NA                    = {na}")
        print(f"  n                     = {n}")
        print(f"  lateral FWHM um       = {fwhm_xy_um:.4f}")
        print(f"  axial FWHM um         = {fwhm_z_um:.4f}")
        print(f"  sigma XY um           = {sigma_xy_um:.4f}")
        print(f"  sigma Z um            = {sigma_z_um:.4f}")
        print(f"  xy_um_per_px          = {xy_um_per_px}")
        print(f"  z_step_um             = {z_step_um}")

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

    psf /= psf.sum() + 1e-12

    return psf.astype(np.float32)
