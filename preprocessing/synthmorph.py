# freesurfer/mri_synthmorph function extension
import os
import numpy as np
import surfa as sf
import tensorflow as tf
import voxelmorph as vxm


# Settings.
default = {
    "model": "joint",
    "hyper": 0.5,
    "extent": 256,
    "steps": 7,
}
choices = {
    "model": ("joint", "deform", "affine", "rigid"),
    "extent": (192, 256),
}
limits = {
    "steps": 5,
}
weights = {
    "joint": (
        "synthmorph.affine.2.h5",
        "synthmorph.deform.2.h5",
    ),
    "deform": ("synthmorph.deform.2.h5",),
    "affine": ("synthmorph.affine.2.h5",),
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
    im : surfa.Volume
        Input image to construct the transform for.
    shape : (3,) array-like
        Spatial shape of the network space.
    center : surfa.Volume, optional
        Center the network space on the center of a reference image.

    Returns
    -------
    out : tuple of (3, 4) NumPy arrays
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
    im : surfa.Volume or NumPy array or TensorFlow tensor
        Input image to transform, without batch dimension.
    trans : array-like
        Transform to apply to the image. A matrix of shape (3, 4), a matrix
        of shape (4, 4), or a displacement field of shape (*space, 3),
        without batch dimension.
    shape : (3,) array-like, optional
        Output shape used for converting matrices to dense transforms. None
        means the shape of the input image will be used.
    normalize : bool, optional
        Min-max normalize the image intensities into the interval [0, 1].
    batch : bool, optional
        Prepend a singleton batch dimension to the output tensor.

    Returns
    -------
    out : float TensorFlow tensor
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
    model : TensorFlow model
        Model to initialize.
    weights : str
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
    trans : (3, 4) or (4, 4) or (*space, 3) array-like
        Matrix transform or displacement field.
    source : castable surfa.ImageGeometry
        Transform source (or moving) image geometry.
    target : castable surfa.ImageGeometry
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


def synthmorph_registration(
    moving,
    fixed,
    interp_method="linear",
    accompanying_images=[],
    m=default["model"],
    o=None,
    O=None,
    H=False,
    t=None,
    T=None,
    i=None,
    j=0,
    g=False,
    r=default["hyper"],
    n=default["steps"],
    e=default["extent"],
    w=[],
    d=None,
):
    in_shape = (e,) * 3
    is_mat = m in ("affine", "rigid")

    if H and not is_mat:
        print("Error: -H is not compatible with deformable registration")
        exit(1)

    if not 0 < r < 1:
        print("Error: regularization strength not in open interval (0, 1)")
        exit(1)

    if n < limits["steps"]:
        print("Error: too few integration steps")
        exit(1)

    # Setup.
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu if g else ""
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if arg_verbose else "3"

    if j:
        tf.config.threading.set_inter_op_parallelism_threads(j)
        tf.config.threading.set_intra_op_parallelism_threads(j)

    mov = sf.load_volume(moving)
    fix = sf.load_volume(fixed)
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
            or not sf.transform.image_geometry_equal(mov.geom, init.source, tol=1e-4)
            or not sf.transform.image_geometry_equal(fix.geom, init.target, tol=1e-4)
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
        fs = os.environ.get("FREESURFER_HOME")
        if not fs:
            sf.system.fatal("set environment variable FREESURFER_HOME or weights")
        w = [os.path.join(fs, "models", f) for f in weights[m]]

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
        out.save(o)

    # Save fixed image registered to moving image.
    if O:
        if H:
            out = fix.copy()
            out.geom.update(vox2world=ras_1 @ fix.geom.vox2world)
        else:
            out = transform(fix, trans=vox_2, shape=mov.shape)
            out = mov.new(out)
        out.save(O)

    for accompanying_image in accompanying_images:
        moving = accompanying_image["moving"]
        out_moving = accompanying_image["out_moving"]
        interp_method = accompanying_image.get("interp_method", "linear")
        mov = sf.load_volume(moving)

        if H:
            out = mov.copy()
            out.geom.update(vox2world=ras_2 @ mov.geom.vox2world)
        else:
            out = transform(
                mov, interp_method=interp_method, trans=vox_1, shape=fix.shape
            )
            out = fix.new(out)
        out.save(out_moving)
