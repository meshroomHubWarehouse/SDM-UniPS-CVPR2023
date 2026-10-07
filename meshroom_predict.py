"""Meshroom API of SDM-UniPS: normal map (and BRDF maps) of one multi-lighting pose.

Used by the mrSDMUniPS Meshroom plugin, whose common layer (psCommon.py) handles everything else (SfMData,
image selection and loading, masks, outputs). Contract shared by the photometric stereo plugins:

    model = loadModel(checkpointPath, useGpu, logger, target=...)
    maps = predict(model, images, mask, **options)

- images: list of float32 RGB arrays (H x W x 3), values as stored in the images (typically [0, 1]),
- mask: bool array (H x W), the pixels of the object (all True without mask),
- maps["normal"]: float32 (H x W x 3) unit normals in the OpenGL camera frame (x right, y up, z towards the
  camera), zero outside the mask and where undefined (normal network),
- maps["albedo"] (H x W x 3), maps["roughness"] and maps["metallic"] (H x W): float32, >= 0, zero outside the
  mask (BRDF network). The albedo is relative: each image is normalized by its own maximum.

The pre/post-processing is the one of the original code (realdata.py, builder.py): crop around the mask with a
margin, extended to a square, resized to a square network input whose side is a multiple of 512, per-image
normalization by the maximum (over the mask) of the mean RGB value, scalable mode, canonical resolution and pixel
samples. Differences with the original code:
- the scalable mode is the default (the whole input at once rarely fits in the GPU memory),
- the images are float arrays: there is no bit-depth division (the per-image normalization removes the scale),
- the crop box includes the last row and column of the mask,
- without a mask (mask all True) the whole image is processed (the original cropped a centered square),
- the maps are mapped back to the full image, multiplied by the input mask, and the BRDF maps are resampled with
  the network mask as weight (no dark fringe along the silhouette),
- the normals are converted from the network frame to the OpenGL camera frame (NATIVE_TO_OPENGL),
- invalid options, checkpoints and inputs raise an error.
"""
import glob
import os

import cv2
import numpy as np
import torch

INTERPOLATIONS = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
TARGETS = ("normal", "brdf", "normal_and_brdf")
# Side of the network input: a multiple of PROCESSING_STEP (also the tile size of the scalable mode)
PROCESSING_STEP = 512
# Canonical resolutions dividing every network input side (multiple of 512); the network was trained at 256
CANONICAL_RESOLUTIONS = (128, 256, 512)
# Sign changes from the frame of the network outputs to the OpenGL camera frame (x right, y up, z towards the
# camera), determined on real data (outward normals along the silhouette, tests/check_real_pose.py of mrSDMUniPS)
NATIVE_TO_OPENGL = np.array([1.0, 1.0, 1.0], np.float32)


class SdmModel:
    """SDM-UniPS networks: the normal network and/or the BRDF network (None when not loaded)."""

    def __init__(self, netNormal, netBrdf, device):
        self.netNormal = netNormal
        self.netBrdf = netBrdf
        self.device = torch.device(device)


def findCheckpoint(checkpointPath, kind):
    """The single *.pytmodel file of <checkpointPath>/<kind> ("normal" or "brdf")."""
    folder = os.path.join(checkpointPath, kind)
    files = sorted(glob.glob(os.path.join(folder, "*.pytmodel")))
    if len(files) != 1:
        raise RuntimeError("SDM-UniPS checkpoint: expected exactly one .pytmodel file in '{}', found {}{}".format(
            folder, len(files), " ({})".format(", ".join(files)) if files else ""))
    return files[0]


def loadNetwork(path, kind, device):
    """Load one SDM-UniPS network (strict: every weight must be in the checkpoint)."""
    from sdm_unips.modules.model import model
    state = torch.load(path, map_location="cpu", weights_only=True)
    # the checkpoints were saved from torch.nn.DataParallel
    state = {key[len("module."):] if key.startswith("module.") else key: value for key, value in state.items()}
    net = model.Net(10000, kind, device).to(device)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("Invalid SDM-UniPS {} checkpoint '{}': {}".format(kind, path, exc)) from exc
    net.no_grad()  # eval mode, no gradients
    return net


def loadModel(checkpointPath, useGpu=True, logger=None, target="normal_and_brdf"):
    """Load the SDM-UniPS networks required by target.

    Args:
        checkpointPath: folder with a normal/ and a brdf/ subfolder, each with one .pytmodel file.
        target: "normal", "brdf" or "normal_and_brdf".

    Raises if a required checkpoint is missing, ambiguous or incomplete.
    """
    if target not in TARGETS:
        raise ValueError("Unknown SDM-UniPS target '{}' (expected one of {})".format(target, ", ".join(TARGETS)))
    if not checkpointPath or not os.path.isdir(checkpointPath):
        raise RuntimeError("SDM-UniPS checkpoint folder not found: '{}'".format(checkpointPath))
    paths = {}
    if "normal" in target:
        paths["normal"] = findCheckpoint(checkpointPath, "normal")
    if "brdf" in target:
        paths["brdf"] = findCheckpoint(checkpointPath, "brdf")
    device = "cuda" if useGpu and torch.cuda.is_available() else "cpu"
    if logger and useGpu and device == "cpu":
        logger.warning("No GPU available: running SDM-UniPS on the CPU (slow).")
    nets = {kind: loadNetwork(path, kind, device) for kind, path in paths.items()}
    if logger:
        logger.info("SDM-UniPS networks on {}: {}".format(device, ", ".join(
            "{} ({})".format(kind, path) for kind, path in paths.items())))
    return SdmModel(nets.get("normal"), nets.get("brdf"), device)


def checkOptions(cropMargin=8, maxProcessingSize=4096, canonicalResolution=256, pixelSamples=10000,
                 scalable=True, outputInterpolation="cubic"):
    """Raise a ValueError if an option of predict is invalid."""
    if int(cropMargin) < 0:
        raise ValueError("cropMargin must be >= 0, got {}".format(cropMargin))
    if int(maxProcessingSize) < PROCESSING_STEP or int(maxProcessingSize) % PROCESSING_STEP:
        raise ValueError("maxProcessingSize must be a multiple of {} (>= {}), got {}".format(
            PROCESSING_STEP, PROCESSING_STEP, maxProcessingSize))
    if int(canonicalResolution) not in CANONICAL_RESOLUTIONS:
        raise ValueError("canonicalResolution must be one of {}, got {}".format(
            ", ".join(str(r) for r in CANONICAL_RESOLUTIONS), canonicalResolution))
    if int(pixelSamples) <= 0:
        raise ValueError("pixelSamples must be > 0, got {}".format(pixelSamples))
    if outputInterpolation not in INTERPOLATIONS:
        raise ValueError("Unknown outputInterpolation '{}' (expected one of {})".format(
            outputInterpolation, ", ".join(INTERPOLATIONS)))


def cropBox(mask, margin):
    """Crop box (r0, r1, c0, c1), ends excluded, as in the original code: the bounding box of the mask extended by
    margin, then extended to a square around its center (clipped to the image); the whole image when the box
    extended by the margin does not fit in the image."""
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("Empty mask")
    r0, r1, c0, c1 = rows[0] - margin, rows[-1] + 1 + margin, cols[0] - margin, cols[-1] + 1 + margin
    if r0 < 0 or c0 < 0 or r1 > height or c1 > width:
        return 0, height, 0, width
    extension = abs((r1 - r0) - (c1 - c0)) // 2
    if r1 - r0 > c1 - c0:
        c0, c1 = max(c0 - extension, 0), min(c1 + extension, width)
    else:
        r0, r1 = max(r0 - extension, 0), min(r1 + extension, height)
    return int(r0), int(r1), int(c0), int(c1)


def processingSize(cropHeight, cropWidth, maxProcessingSize):
    """Side of the square network input: max(512, min(maxProcessingSize, floor(crop side / 512) * 512))."""
    side = (max(cropHeight, cropWidth) // PROCESSING_STEP) * PROCESSING_STEP
    return int(max(PROCESSING_STEP, min(maxProcessingSize, side)))


def prepareInputs(images, mask, box, size):
    """Crop, resize to size x size (bicubic) and normalize the images as the original code.

    Returns:
        (images float32 N x size x size x 3, network mask bool size x size)
    """
    r0, r1, c0, c1 = box
    netMask = cv2.resize(mask[r0:r1, c0:c1].astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC) > 0.5
    if not netMask.any():
        raise RuntimeError("The mask is empty at the network resolution ({0}x{0})".format(size))
    inputs = np.empty((len(images), size, size, 3), np.float32)
    for i, image in enumerate(images):
        crop = np.ascontiguousarray(np.asarray(image, np.float32)[r0:r1, c0:c1])
        inputs[i] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    # bicubic overshoot: the original code resized integer images, saturated at 0
    np.maximum(inputs, 0.0, out=inputs)
    # per-image normalization by the maximum, over the mask, of the mean RGB value
    brightest = inputs.mean(axis=3)[:, netMask].max(axis=1)
    inputs /= (brightest + 1.0e-6)[:, None, None, None]
    return inputs, netMask


def forward(net, images, mask, nbImages, resolution, canonicalResolution):
    """One network evaluation (images: B x 3 x R x R x N, mask: B x 1 x R x R)."""
    ones = torch.ones(images.shape[0], 1)
    return net(images, mask, nbImages.reshape(-1, 1), decoder_resolution=resolution * ones,
               canonical_resolution=canonicalResolution * ones)


def runNetwork(net, images, mask, kind, canonicalResolution, scalable):
    """Run a network on the whole input (scalable: on 512 x 512 interleaved sub-grids, as the original code).

    Args:
        images: 1 x 3 x S x S x N tensor, mask: 1 x 1 x S x S tensor.
        kind: "normal" (returns [normal]) or "brdf" (returns [albedo, roughness, metallic]).

    Returns:
        list of C x S x S CPU tensors, zero outside the mask.
    """
    from sdm_unips.modules.model.decompose_tensors import divide_tensor_spatial, merge_tensor_spatial
    batch, channels, height, width, nbImages = images.shape
    nbImagesArray = torch.tensor([nbImages], dtype=torch.long)

    def select(outputs):
        return [outputs[0]] if kind == "normal" else list(outputs[1:])

    if not scalable:
        outputs = select(forward(net, images, mask, nbImagesArray, height, canonicalResolution))
        return [(output * mask)[0].float().cpu() for output in outputs]

    tile = PROCESSING_STEP
    tilesImages = divide_tensor_spatial(images.permute(0, 4, 1, 2, 3).reshape(-1, channels, height, width),
                                        block_size=tile, method="tile_stride")
    tilesImages = tilesImages.reshape(batch, nbImages, -1, channels, tile, tile).permute(0, 2, 3, 4, 5, 1)
    tilesMask = divide_tensor_spatial(mask, block_size=tile, method="tile_stride")
    outputChannels = [3] if kind == "normal" else [3, 1, 1]
    tiles = [[] for _ in outputChannels]
    for k in range(tilesImages.shape[1]):
        tileMask = tilesMask[:, k]
        if torch.sum(tileMask) > 0:
            outputs = select(forward(net, tilesImages[:, k], tileMask, nbImagesArray, tile, canonicalResolution))
            for maps, output in zip(tiles, outputs):
                maps.append((output * tileMask).float().cpu())
        else:
            for maps, nbChannels in zip(tiles, outputChannels):
                maps.append(torch.zeros((batch, nbChannels, tile, tile)))
    return [merge_tensor_spatial(torch.stack(maps, dim=1).permute(1, 0, 2, 3, 4), method="tile_stride")[0]
            for maps in tiles]


def resize(values, size, interpolation):
    """cv2.resize of a H x W x C array (C = 1 kept), size = (width, height)."""
    resized = cv2.resize(np.ascontiguousarray(values), size, interpolation=interpolation)
    return resized.reshape(size[1], size[0], -1)


def toImage(values, box, shape):
    """Paste a crop (h x w x C) into a zero image of shape (H, W); returns H x W x C."""
    r0, r1, c0, c1 = box
    image = np.zeros(tuple(shape) + (values.shape[2],), np.float32)
    image[r0:r1, c0:c1] = values
    return image


def normalToImage(normal, box, mask, interpolation):
    """Network normals (S x S x 3) to the full image: resampling to the crop size, unit normals where the
    resampled norm is within [0.5, 1.5] (original post-processing), zero elsewhere and outside the mask."""
    r0, r1, c0, c1 = box
    normal = resize(normal, (c1 - c0, r1 - r0), interpolation)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    valid = np.abs(1.0 - norm) < 0.5
    normal = np.where(valid, normal / np.maximum(norm, 1.0e-12), 0.0)
    return toImage(normal, box, mask.shape) * mask[:, :, None]


def brdfToImage(values, netMask, box, mask, interpolation):
    """Network BRDF map (S x S x C) to the full image, resampled with the network mask as weight (the values
    outside the network mask are zero, a plain resampling would darken the silhouette), clipped to >= 0, zero
    outside the mask."""
    r0, r1, c0, c1 = box
    size = (c1 - c0, r1 - r0)
    # same (linear) resampling of the weight: a constant map is resampled exactly, overshoot included
    weight = resize(netMask.astype(np.float32), size, interpolation)
    values = resize(values, size, interpolation)
    values = np.where(weight > 0.01, values / np.maximum(weight, 0.01), 0.0)
    return toImage(np.maximum(values, 0.0), box, mask.shape) * mask[:, :, None]


def predict(model, images, mask, cropMargin=8, maxProcessingSize=4096, canonicalResolution=256, pixelSamples=10000,
            scalable=True, outputInterpolation="cubic"):
    """Maps of one pose: "normal" (normal network) and "albedo", "roughness", "metallic" (BRDF network).

    Args:
        cropMargin: margin (pixels) around the bounding box of the mask; the whole image is used when the box is
            closer than this margin to the image border.
        maxProcessingSize: maximum side of the square network input (multiple of 512). The crop is resized to
            max(512, min(maxProcessingSize, floor(crop side / 512) * 512)).
        canonicalResolution: resolution of the global image encoder (one of CANONICAL_RESOLUTIONS, trained at 256).
        pixelSamples: number of pixels decoded together by the pixel-sampling transformer.
        scalable: process the input as 512 x 512 interleaved sub-grids, one at a time (default). Otherwise the whole
            input is processed at once (original default), which needs much more GPU memory: the encoder processes
            N x (side / canonicalResolution)^2 tiles at once (e.g. 9 images at 1536 do not fit in 16 GB).
        outputInterpolation: resampling of the prediction back to the crop size ("area", "linear", "cubic").
    """
    checkOptions(cropMargin, maxProcessingSize, canonicalResolution, pixelSamples, scalable, outputInterpolation)
    if model.netNormal is None and model.netBrdf is None:
        raise RuntimeError("No SDM-UniPS network loaded")
    if not images:
        raise ValueError("No input image")
    mask = np.asarray(mask, bool)
    for image in images:
        if image.shape != mask.shape + (3,):
            raise ValueError("Image of shape {} for a mask of shape {}".format(image.shape, mask.shape))

    box = cropBox(mask, int(cropMargin))
    size = processingSize(box[1] - box[0], box[3] - box[2], int(maxProcessingSize))
    inputs, netMask = prepareInputs(images, mask, box, size)
    interpolation = INTERPOLATIONS[outputInterpolation]

    device = model.device
    # 1 x 3 x S x S x N images and 1 x 1 x S x S mask
    imagesTensor = torch.from_numpy(np.ascontiguousarray(inputs.transpose(3, 1, 2, 0)))[None].to(device)
    maskTensor = torch.from_numpy(netMask.astype(np.float32))[None, None].to(device)
    del inputs

    maps = {}
    with torch.no_grad():
        if model.netNormal is not None:
            model.netNormal.pixel_samples = int(pixelSamples)
            normal, = runNetwork(model.netNormal, imagesTensor, maskTensor, "normal", int(canonicalResolution),
                                 scalable)
            normal = normalToImage(normal.permute(1, 2, 0).numpy(), box, mask, interpolation)
            maps["normal"] = (normal * NATIVE_TO_OPENGL).astype(np.float32)
        if model.netBrdf is not None:
            model.netBrdf.pixel_samples = int(pixelSamples)
            brdf = runNetwork(model.netBrdf, imagesTensor, maskTensor, "brdf", int(canonicalResolution), scalable)
            for name, values in zip(("albedo", "roughness", "metallic"), brdf):
                values = brdfToImage(values.permute(1, 2, 0).numpy(), netMask, box, mask, interpolation)
                maps[name] = values if name == "albedo" else values[:, :, 0]
    return maps
