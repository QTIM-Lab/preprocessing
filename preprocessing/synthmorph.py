"""
The `synthmorph` module uses the Synthmorph models to perform image registration.

Public Functions
________________

synthmorph_registration
"""
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
import voxelmorph as vxm
import surfa as sf

from SimpleITK import Image
from preprocessing.utils import sitk_to_surfa, surfa_to_sitk, check_for_models
from preprocessing.constants import PREPROCESSING_MODELS_PATH
from typing import Dict, Literal, Sequence

check_for_models(PREPROCESSING_MODELS_PATH)

weights = {
    "joint": (
        "synthmorph.affine.2.h5",
        "synthmorph.deform.2.h5",
    ),
    "deform": ("synthmorph.deform.3.h5",),
    "affine": ("synthmorph.affine.2.h5",),
    "affine_crop": ("synthmorph.affine.crop.h5",),
    "rigid": ("synthmorph.rigid.1.h5",),
}


def network_space(im, shape, center=None):
    """Construct transform from network space to the voxel space of an image.

    Constructs a coordinate transform from the space the network will operate
    in to the zero-based image index space. The network space has isotropic
    1-mm voxels, left-inferior-anterior (LIA) orientation, and no shear. It is
    centered on the field of view, or that of a reference image. This space is
    an indexed voxel space, not world space.

    Parameters
    ----------
    im: surfa.Volume
        Input image to construct the transform for.

    shape: (3,) array-like
        Spatial shape of the network space.

    center: surfa.Volume, optional
        Center the network space on the center of a reference image.

    Returns
    -------
    out: tuple of (3, 4) NumPy arrays
        Transform from network to input-image space and its inverse, thinking
        coordinates.

    """
    old = im.geom
    new = sf.ImageGeometry(
        shape=shape,
        voxsize=1,
        rotation="LIA",
        center=old.center if center is None else center.geom.center,
        shear=None,
    )

    net_to_vox = old.world2vox @ new.vox2world
    vox_to_net = new.world2vox @ old.vox2world
    return np.float32(net_to_vox.matrix), np.float32(vox_to_net.matrix)


def transform(
    im, trans, interp_method="linear", shape=None, normalize=False, batch=False
):
    """Apply a spatial transform to 3D image voxel data in dimensions.

    Applies a transformation matrix operating in zero-based index space or a
    displacement field to an image buffer.

    Parameters
    ----------
    im: surfa.Volume or NumPy array or TensorFlow tensor
        Input image to transform, without batch dimension.

    trans: array-like
        Transform to apply to the image. A matrix of shape (3, 4), a matrix
        of shape (4, 4), or a displacement field of shape (*space, 3),
        without batch dimension.

    shape: (3,) array-like, optional
        Output shape used for converting matrices to dense transforms. None
        means the shape of the input image will be used.

    normalize: bool, optional
        Min-max normalize the image intensities into the interval [0, 1].

    batch: bool, optional
        Prepend a singleton batch dimension to the output tensor.

    Returns
    -------
    out: float TensorFlow tensor
        Transformed image with with a trailing feature dimension.

    """
    # Add singleton feature dimension if needed.
    if tf.rank(im) == 3:
        im = im[..., tf.newaxis]

    out = vxm.utils.transform(
        im,
        trans,
        interp_method=interp_method,
        fill_value=0,
        shift_center=False,
        shape=shape,
    )

    if normalize:
        out -= tf.reduce_min(out)
        out /= tf.reduce_max(out)

    if batch:
        out = out[tf.newaxis, ...]

    return out


def load_weights(model, weights):
    """Load weights into model or submodel.

    Attempts to load weights into a model or its direct submodels and return on
    first success. Raises a `ValueError` if unsuccessful.

    Parameters
    ----------
    model: TensorFlow model
        Model to initialize.

    weights: str
        Path to weights file.

    """
    cand = (model, *(f for f in model.layers if isinstance(f, tf.keras.Model)))
    for c in cand:
        try:
            c.load_weights(weights)
            return
        except ValueError as e:
            if c is cand[-1]:
                raise e


def convert_to_ras(trans, source, target):
    """Convert a voxel-to-voxel transform to a world-to-world (RAS) transform.

    For input displacement fields, we want to output shifts that we can add to
    the RAS coordinates corresponding to indices (0, 1, ...) along each axis in
    the target voxel space - instead of adding the shifts to discrete RAS
    indices (0, 1, ...) along each axis. Therefore, we start with target voxel
    coordinates and subtract the corresponding RAS coordinates at the end.
    Naming transforms after their effect on coordinates,

        out = mov_to_ras @ fix_to_mov + x_fix - x_ras
        out = mov_to_ras @ fix_to_mov + x_fix - fix_to_ras @ x_fix
        out = mov_to_ras @ fix_to_mov + (identity - fix_to_ras) @ x_fix

    Parameters
    ----------
    trans: (3, 4) or (4, 4) or (*space, 3) array-like
        Matrix transform or displacement field.

    source: castable surfa.ImageGeometry
        Transform source (or moving) image geometry.

    target: castable surfa.ImageGeometry
        Transform target (or fixed) image geometry.

    Returns
    -------
    out: float TensorFlow tensor
        Converted world-space transform of shape (3, 4) if `trans` is a matrix
        or (*space, 3) if it is a displacement field.

    """
    mov = sf.transform.geometry.cast_image_geometry(source)
    fix = sf.transform.geometry.cast_image_geometry(target)
    mov_to_ras = np.float32(mov.vox2world.matrix)
    fix_to_ras = np.float32(fix.vox2world.matrix)
    ras_to_fix = np.float32(fix.world2vox.matrix)
    prop = dict(shift_center=False, shape=fix.shape)

    # Simple matrix multiplication.
    if vxm.utils.is_affine_shape(trans.shape):
        return vxm.utils.compose((mov_to_ras, trans, ras_to_fix), **prop)

    # Target voxel coordinate grid.
    x_fix = (tf.range(x, dtype=tf.float32) for x in fix.shape)
    x_fix = tf.meshgrid(*x_fix, indexing="ij")
    x_fix = tf.stack(x_fix)
    x_fix = tf.reshape(x_fix, shape=(3, -1))

    # We go from target voxels to source voxels to RAS, then subtract the RAS
    # coordinates corresponding to the target voxels we started at.
    mat = tf.eye(4) - fix_to_ras
    x = mat[:-1, -1:] + mat[:-1, :-1] @ x_fix
    x = tf.transpose(x)
    x = tf.reshape(x, shape=(*fix.shape, -1))
    return vxm.utils.compose((mov_to_ras, trans), **prop) + x

# TODO consider support for using transforms as inputs for t,T,I.
# In the form of the surfa transform
# or wrapped with the sitk converters for sitk transforms
def synthmorph_registration(
    moving: str | Image,
    fixed: str | Image,
    interp_method: Literal["linear", "nearest"] = "linear",
    accompanying_images: Sequence[Dict[str, str]] = [],
    sitk_out: bool = True,
    sitk_im_cache: Dict[str, Image] = {},
    accompanying_in_cache: bool = False,
    m: Literal["joint", "deform", "affine", "affine_crop", "rigid"] = "joint",
    o: str | None = None,
    O: str | None = None,
    H: bool = False,
    t: str | None = None,
    T: str | None = None,
    i: str | None = None,
    j: int = 0,
    r: float = 0.5,
    n: int = 7,
    e: Literal[192, 256] = 256,
    w: Sequence[str] = [],
    d: str | None = None,
) -> Dict[str, Image]:
    """
    Adaptation of Freesurfer's mri_synthmorph command that allows transformation of
    multiple files at once.

    Parameters
    __________
    moving: str | Image
        The moving image to be used in registration. Will accept a string indicating
        the location of the moving image NIfTI file or the corresponding SimpleITK.Image.

    fixed: str | Image
        The fixed image to be used in registration. Will accept a string indicating
        the location of the fixed image NIfTI file or the corresponding SimpleITK.Image.

    interp_method: Literal["linear", "nearest"]
        The interpolation method to be used with the derived transformation. Defaults to
        "linear".

    accompanying_images: Sequence[Dict[str, str]]
        Accompanying images that share the space of the moving image. The derived transform
        will be applied to all of these images. `accompanying_images` is provided as a
        sequence of dictionaries with the form:
            [
                {
                    "moving": "/path/to/moving.nii.gz",
                    "moved": "/path/to/moved.nii.gz",
                    "interp_method": Literal["llinear", "nearest"] = "linear"
                },
                ...
            ]

    sitk_out: bool
        Whether the transformed images will be output as a SimpleITK.Image. If True,
        `sitk_im_cache` will be updated to contain the moved images. Otherwise, the
        moved images will be written to the specified file path. Defaults to True.

    sitk_im_cache: Dict[str, Image]
        A dictionary mapping file locations (as strings) to SimpleITK.Images.

    accompanying_in_cache: bool
        Whether all of the paths within `accompanying_images` are stored within `sitk_im_cache`.
        Defaults to False.

    m: Literal["joint", "deform", "affine", "affine_crop", "rigid"]
        Model weights used for registration. Defaults to "joint".

    o: str | None
        Save path for the moving image registered to the fixed image. Defaults to None.

    O: str | None
        Save path for the fixed image registered to the moving image. Defaults to None.

    H: bool
        Update the voxel-to-world matrix instead of resampling when saving images to
        `o` or `O`. For matrix transforms only. Not all software supports headers with
        shear from affine registration. Defaults to False.

    t: str | None
        Save path for the transform from the moving to the fixed image, including any
        initialization. Defaults to None.

    T: str | None
        Save path for the transform from the fixed to the moving image, including any
        initialization. Defaults to None.

    i: str | None
        Path to an initial matrix transform to the moving image before the registration.

    j: int
        Number of TensorFlow threads. Defaults to 0

    r: float
        Regularization parameter in the open interval (0, 1) for deformable registration.
        Higher values lead to smoother warps. Defaults to 0.5.

    n: int
        Integration steps for deformable registration. Lower numbers improve speed and
        memory use but can lead to inaccuracies and folding voxels. Defaults to 7. Should
        not be less than 5.

    e: Literal[192, 256]
        Isotropic extent of the registration space in unit voxels. Lower values improve
        speed and memory use but may crop the anatomy of interest. Defaults to 256.

    w: Sequence[str]
        Alternative weights for the registration model. Provide more than one to set
        submodel weights.

    d: str | None
        An output directory which will contain the input images moved to network space, named
        "inp1.mgz" and "inp2.mgz".

    Returns
    _______
    sitk_im_cache: Dict[str, Image]
        A potentially updated version of the input `sitk_im_cache`, which contains the registered
        images if `sitk_out` is True.
    """
    in_shape = (e,) * 3
    is_mat = m in ("affine", "affine_crop", "rigid")

    if H and not is_mat:
        print("Error: -H is not compatible with deformable registration")
        exit(1)

    if not 0 < r < 1:
        print("Error: regularization strength not in open interval (0, 1)")
        exit(1)

    if n < 5:
        print("Error: too few integration steps")
        exit(1)

    with tf.device("/CPU:0"):
        if j:
            tf.config.threading.set_inter_op_parallelism_threads(j)
            tf.config.threading.set_intra_op_parallelism_threads(j)

        if isinstance(moving, Image):
            # print(f"moving: {moving.GetDirection(), moving.GetSize()}")
            mov = sitk_to_surfa(moving)

        else:
            mov = sf.load_volume(moving)

        if isinstance(fixed, Image):
            # print(f"fixed: {fixed.GetDirection(), fixed.GetSize()}")
            fix = sitk_to_surfa(fixed)

        else:
            fix = sf.load_volume(fixed)

        # pdb.set_trace()

        if not len(mov.shape) == len(fix.shape) == 3:
            sf.system.fatal("input images are not single-frame volumes")

        # Transforms between native voxel and network coordinates. Voxel and network
        # spaces differ for each image. The networks expect isotropic 1-mm LIA spaces.
        # We center these on the original images, except for deformable registration:
        # this assumes prior affine registration, so we center the moving network space
        # on the fixed image, to take into account affine transforms applied by
        # resampling, updating the header, or passed on the command line alike.
        center = fix if m == "deform" else None
        net_to_mov, mov_to_net = network_space(mov, shape=in_shape, center=center)
        net_to_fix, fix_to_net = network_space(fix, shape=in_shape)

        # Coordinate transforms from and to world space. There is only one world.
        mov_to_ras = np.float32(mov.geom.vox2world.matrix)
        fix_to_ras = np.float32(fix.geom.vox2world.matrix)
        ras_to_mov = np.float32(mov.geom.world2vox.matrix)
        ras_to_fix = np.float32(fix.geom.world2vox.matrix)

        # Incorporate an initial matrix transform. It maps from fixed to moving world
        # coordinates, so we start with fixed network space on the right. FreeSurfer
        # LTAs store the inverse of the transform.
        if i:
            init = sf.load_affine(i).convert(space="world")
            if (
                init.ndim != 3
                or not sf.transform.image_geometry_equal(
                    mov.geom, init.source, tol=1e-4
                )
                or not sf.transform.image_geometry_equal(
                    fix.geom, init.target, tol=1e-4
                )
            ):
                sf.system.fatal("initial transform geometry does not match images")

            net_to_mov = np.float32(ras_to_mov @ init.inv() @ fix_to_ras @ net_to_fix)
            mov_to_net = np.float32(fix_to_net @ ras_to_fix @ init @ mov_to_ras)

        # Take the input images to network space. When saving the moving image with the
        # correct voxel-to-RAS matrix after incorporating an initial matrix transform,
        # an image viewer taking this matrix into account will show an unchanged image.
        # However, the networks only see the voxel data, which have been moved.
        inputs = (
            transform(
                mov,
                net_to_mov,
                interp_method=interp_method,
                shape=in_shape,
                normalize=True,
                batch=True,
            ),
            transform(
                fix,
                net_to_fix,
                interp_method=interp_method,
                shape=in_shape,
                normalize=True,
                batch=True,
            ),
        )
        if d:
            os.makedirs(d, exist_ok=True)
            inp_1 = os.path.join(d, "inp_1.mgz")
            inp_2 = os.path.join(d, "inp_2.mgz")
            geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
            geom_2 = sf.ImageGeometry(in_shape, vox2world=fix_to_ras @ net_to_fix)

            if sitk_out:
                sitk_im_cache[inp_1] = surfa_to_sitk(sf.Volume(inputs[0][0], geom_1))
                sitk_im_cache[inp_2] = surfa_to_sitk(sf.Volume(inputs[1][0], geom_2))

            else:
                sf.Volume(inputs[0][0], geom_1).save(inp_1)
                sf.Volume(inputs[1][0], geom_2).save(inp_2)

        # Network.
        prop = dict(in_shape=in_shape, bidir=True)
        if is_mat:
            prop.update(make_dense=False, rigid=m == "rigid")
            model = vxm.networks.VxmAffineFeatureDetector(**prop)

        else:
            prop.update(mid_space=True, int_steps=n, skip_affine=m == "deform")
            model = vxm.networks.HyperVxmJoint(**prop)
            inputs = (np.asarray([r]), *inputs)

        # Weights.
        if not w:
            w = [os.path.join(PREPROCESSING_MODELS_PATH, f) for f in weights[m]]

        for f in w:
            load_weights(model, weights=f)

        # Inference. The first transform maps from the moving to the fixed image, or
        # equivalently, from fixed to moving coordinates. The second is the inverse.
        vox_1, vox_2 = map(tf.squeeze, model(inputs))

        # Convert transforms between moving and fixed network spaces to transforms
        # between the original voxel spaces. Also compute transforms operating in RAS.
        prop = dict(shift_center=False, shape=fix.shape)
        vox_1 = vxm.utils.compose((net_to_mov, vox_1, fix_to_net), **prop)
        ras_1 = convert_to_ras(vox_1, source=mov, target=fix)

        prop = dict(shift_center=False, shape=mov.shape)
        vox_2 = vxm.utils.compose((net_to_fix, vox_2, mov_to_net), **prop)
        ras_2 = convert_to_ras(vox_2, source=fix, target=mov)

        # Save transform from moving to fixed image. FreeSurfer LTAs store the inverse.
        if t:
            if is_mat:
                out = sf.Affine(ras_2, source=mov, target=fix, space="world")
            else:
                out = fix.new(ras_1)
            out.save(t)

        # Save transform from fixed to moving image. FreeSurfer LTAs store the inverse.
        if T:
            if is_mat:
                out = sf.Affine(ras_1, source=fix, target=mov, space="world")
            else:
                out = mov.new(ras_2)
            out.save(T)

        # Save moving image registered to fixed image.
        if o:
            if H:
                out = mov.copy()
                out.geom.update(vox2world=ras_2 @ mov.geom.vox2world)
            else:
                out = transform(
                    mov, interp_method=interp_method, trans=vox_1, shape=fix.shape
                )
                out = fix.new(out)

            if sitk_out:
                sitk_im_cache[o] = surfa_to_sitk(out)

            else:
                out.save(o)

        # Save fixed image registered to moving image.
        if O:
            if H:
                out = fix.copy()
                out.geom.update(vox2world=ras_1 @ fix.geom.vox2world)
            else:
                out = transform(fix, trans=vox_2, shape=mov.shape)
                out = mov.new(out)

            if sitk_out:
                sitk_im_cache[O] = surfa_to_sitk(out)

            else:
                out.save(O)

        for accompanying_image in accompanying_images:
            moving = accompanying_image["moving"]
            moved = accompanying_image["moved"]
            interp_method = accompanying_image.get("interp_method", "linear")

            if accompanying_in_cache:
                mov = sitk_to_surfa(sitk_im_cache[moving])
            else:
                mov = sf.load_volume(moving)

            if H:
                out = mov.copy()
                out.geom.update(vox2world=ras_2 @ mov.geom.vox2world)
            else:
                out = transform(
                    mov, interp_method=interp_method, trans=vox_1, shape=fix.shape
                )
                out = fix.new(out)

            if sitk_out:
                sitk_im_cache[moved] = surfa_to_sitk(out)

            else:
                out.save(moved)

    return sitk_im_cache


__all__ = ["synthmorph_registration"]
