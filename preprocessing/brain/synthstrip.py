"""
The `synthmstrip` module uses the Synthmorph models to perform image registration.

Public Functions
----------------
synthstrip_skullstrip
    Adaptation of Freesurfer's mri_synthstrip command. One of `out`, `m`, or `d` must
    be specified.
"""
import os
import torch
import numpy as np
import surfa as sf

from torch import nn
from SimpleITK import Image
from preprocessing.utils import sitk_to_surfa, surfa_to_sitk, check_for_models
from preprocessing.constants import PREPROCESSING_MODELS_PATH
from typing import Dict

check_for_models(PREPROCESSING_MODELS_PATH)


def extend_sdt(sdt, border=1):
    """Extend SynthStrip's narrow-band signed distance transform (SDT).

    Recompute the positive outer part of the SDT estimated by SynthStrip, for
    borders that likely exceed the 4-5 mm band. Keeps the negative inner part
    intact and only computes the outer part where needed to save time.

    Parameters
    ----------
    sdt: sf.Volume
        Narrow-band signed distance transform estimated by SynthStrip.

    border: float, optional
        Mask border threshold in millimeters.

    Returns
    -------
    sdt: sf.Volume
        Extended SDT.

    """
    if border < int(sdt.max()):
        return sdt

    # Find bounding box.
    mask = sdt < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)

    # Add requested border.
    gap = int(border + 0.5)
    low = (max(i - gap, 0) for i in low)
    upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))

    # Compute EDT within bounding box. Keep interior values.
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt, fill_value=100)
    out[ind] = sf.Volume(mask[ind]).distance()
    out[keep] = sdt[keep]

    return sdt.new(out)


class StripModel(nn.Module):
    def __init__(
        self,
        nb_features=16,
        nb_levels=7,
        feat_mult=2,
        max_features=64,
        nb_conv_per_level=2,
        max_pool=2,
        return_mask=False,
    ):
        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation="leaky"):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == "leaky":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == None:
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


def synthstrip_skullstrip(
    image: str | Image,
    sitk_im_cache: Dict[str, Image] = {},
    sitk_out: bool = True,
    out: str | None = None,
    m: str | None = None,
    d: str | None = None,
    b: int = 1,
    threads: int | None = None,
    no_csf: bool = False,
    mod: str | None = None,
) -> Dict[str, Image]:
    """
    Adaptation of Freesurfer's mri_synthstrip command. One of `out`, `m`, or `d` must
    be specified.

    Parameters
    ----------
    image: str | Image
        The image that will be used for skullstripping. Will accept a string indicating
        the location of the moving image NIfTI file or the corresponding SimpleITK.Image.

    sitk_im_cache: Dict[str, Image]
        A dictionary mapping file locations (as strings) to SimpleITK.Images.

    sitk_out: bool
        Whether the transformed images will be output as a SimpleITK.Image. If True,
        `sitk_im_cache` will be updated to contain the moved images. Otherwise, the
        moved images will be written to the specified file path. Defaults to True.

    out: str | None
        Save path to the skullstripped image. Defaults to None.

    m: str | None
        Save path to the binary brain mask. Defaults to None.

    d: str | None
        Save path to the distance transform. Defaults to None.

    b: int
        Mask border threshold in mm. Defaults to 1.

    no_csf: bool
        Whether to exclude CSF from brain border. Defaults to False.

    mod: str | None
        Alternative weights for the skullstrip model.

    Returns
    -------
    sitk_im_cache: Dict[str, Image]
        A potentially updated version of the input `sitk_im_cache`, which contains the registered
        images if `sitk_out` is True.
    """
    # sanity check on the inputs
    if not out and not m and not d:
        sf.system.fatal("Must provide at least one -o, -m, or -d output flag.")

    device = torch.device("cpu")
    if threads is not None:
        torch.set_num_threads(threads)

    with torch.no_grad():
        model = StripModel()
        model.to(device)
        model.eval()

    # load model weights
    if mod is not None:
        modelfile = mod
        # print("Using custom model weights")
    else:
        version = "1"

        if no_csf:
            # print("Excluding CSF from brain boundary")
            modelfile = os.path.join(
                PREPROCESSING_MODELS_PATH, f"synthstrip.nocsf.{version}.pt"
            )
        else:
            modelfile = os.path.join(
                PREPROCESSING_MODELS_PATH, f"synthstrip.{version}.pt"
            )
    checkpoint = torch.load(modelfile, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # load input volume

    if isinstance(image, Image):
        image = sitk_to_surfa(image)

    else:
        image = sf.load_volume(image)
    # print(f"Input image read from: {args.image}")

    # loop over frames (try not to keep too much data in memory)
    # print(f"Processing frame (of {image.nframes}):", end=" ", flush=True)
    dist = []
    mask = []
    for f in range(image.nframes):
        # print(f + 1, end=" ", flush=True)
        frame = image.new(image.framed_data[..., f])

        # conform, fit to shape with factors of 64
        conformed = frame.conform(
            voxsize=1.0, dtype="float32", method="nearest", orientation="LIA"
        ).crop_to_bbox()
        target_shape = np.clip(
            np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320
        )
        conformed = conformed.reshape(target_shape)

        # normalize
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)

        # predict the sdt
        with torch.no_grad():
            input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(
                device
            )
            sdt = model(input_tensor).cpu().numpy().squeeze()

        # extend the sdt if needed, unconform
        sdt = extend_sdt(conformed.new(sdt), border=b)
        sdt = sdt.resample_like(image, fill=100)
        dist.append(sdt)

        # extract mask, find largest CC to be safe
        mask.append((sdt < b).connected_component_mask(k=1, fill=True))

    # combine frames and end line
    dist = sf.stack(dist)
    mask = sf.stack(mask)

    # write the masked output
    if out:
        image[mask == 0] = np.min([0, image.min()])

        if sitk_out:
            sitk_im_cache[out] = surfa_to_sitk(image)

        else:
            image.save(out)

    # write the brain mask
    if m:
        if sitk_out:
            sitk_im_cache[m] = surfa_to_sitk(image.new(mask))

        else:
            image.new(mask).save(m)

    # write the distance transform
    if d:
        if sitk_out:
            sitk_im_cache[d] = surfa_to_sitk(image.new(dist))

        else:
            image.new(dist).save(d)

    return sitk_im_cache


__all__ = ["synthstrip_skullstrip"]
