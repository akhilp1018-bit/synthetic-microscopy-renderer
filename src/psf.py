import numpy as np
import tifffile


def _move_psf_to_zyx(arr: np.ndarray) -> np.ndarray:
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
    pz, py, px = map(int, shape_zyx)

    lam_um = lambda_nm * 1e-3

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