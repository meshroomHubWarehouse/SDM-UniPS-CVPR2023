"""
SfM-based inference for SDM-UniPS.

Reads an AliceVision SfMData JSON (or .sfm via pyalicevision), runs
photometric stereo per pose using SDM-UniPS, and produces output maps
(normals, albedo, roughness, metallic) plus a JSON mapping poseIds to
output paths.

The crop/resize/normalize preprocessing faithfully reproduces the logic
from realdata.py so that the neural network receives correctly formatted
input.
"""

import argparse
import glob
import json
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SfM loading: pyalicevision first, JSON fallback
# ---------------------------------------------------------------------------

def load_sfm(sfm_path):
    """Load SfMData — try pyalicevision first (supports .sfm, .abc, .json),
    fallback to json.load.  Uses sfmDataIO.save() to a temp JSON so ALL
    fields (version, intrinsics, metadata, surveys…) are preserved."""
    try:
        from pyalicevision import sfmData as avSfmData, sfmDataIO
        import tempfile
        data = avSfmData.SfMData()
        if sfmDataIO.load(data, sfm_path, sfmDataIO.ALL):
            logger.info("Loaded SfMData via pyalicevision: %s", sfm_path)
            with tempfile.NamedTemporaryFile(suffix=".sfm", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sfmDataIO.save(data, tmp_path, sfmDataIO.ALL)
                with open(tmp_path, "r") as f:
                    return json.load(f)
            finally:
                os.unlink(tmp_path)
        logger.warning("pyalicevision failed to load %s, falling back to JSON", sfm_path)
    except ImportError:
        logger.info("pyalicevision not available, using JSON loader")
    with open(sfm_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# View grouping and image/mask loading
# ---------------------------------------------------------------------------

def group_views_by_pose(sfm_data):
    """Group views by poseId.

    Returns:
        dict mapping poseId (str) -> list of view dicts
    """
    groups = {}
    for view in sfm_data.get("views", []):
        pose_id = str(view.get("poseId", view.get("viewId")))
        groups.setdefault(pose_id, []).append(view)
    return groups


def load_images_for_pose(views, nb_img=10, downscale=1):
    """Load images for one pose group.

    Args:
        views: list of view dicts sharing the same poseId
        nb_img: max number of images to use (-1 = all)
        downscale: integer downscale factor

    Returns:
        list of numpy arrays (H, W, 3) in original dtype (uint8 or uint16)
    """
    image_paths = []
    for v in views:
        path = v.get("path", "")
        if path and os.path.isfile(path):
            image_paths.append(path)
        else:
            logger.warning("Image not found for viewId %s: %s",
                           v.get("viewId"), path)

    if not image_paths:
        raise RuntimeError(
            f"No valid images for pose {views[0].get('poseId')}")

    if nb_img > 0 and nb_img < len(image_paths):
        indices = np.random.choice(len(image_paths), nb_img, replace=False)
        image_paths = [image_paths[i] for i in sorted(indices)]

    imgs = []
    for p in image_paths:
        img = cv2.cvtColor(
            cv2.imread(p, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
            cv2.COLOR_BGR2RGB,
        )
        if downscale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downscale, h // downscale),
                             interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return imgs


def extract_alpha_mask(views):
    """Extract mask by ANDing all alpha channels from the pose's images.

    Images with all-white alpha are skipped. The result is the intersection
    of all non-trivial alpha masks (logical AND), keeping only the object area.

    Returns a 2D float32 array (H, W) with 0/1 values, or None.
    """
    combined = None
    count = 0
    for v in views:
        path = v.get("path", "")
        if not path or not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or len(img.shape) != 3 or img.shape[2] != 4:
            continue
        alpha = img[:, :, 3]
        # Skip all-white (trivial) alpha channels
        if alpha.min() > 250:
            continue
        mask = (alpha > 0).astype(np.float32)
        if combined is None:
            combined = mask
        else:
            combined = combined * mask  # logical AND
        count += 1
    if combined is not None:
        logger.info("Extracted alpha mask from %d images (AND)", count)
    return combined


def find_mask_for_pose(pose_id, mask_folder, view_ids=None, views=None):
    """Find a mask image for a given pose.

    Search order: mask_folder files, then alpha channel from images.
    Returns a 2D float32 array (H, W) with 0/1 values, or None.
    """
    if mask_folder and os.path.isdir(mask_folder):
        for candidate_id in [pose_id] + (view_ids or []):
            path = os.path.join(mask_folder, f"{candidate_id}.png")
            if os.path.isfile(path):
                mask = cv2.imread(
                    path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                mask = (mask > 0).astype(np.float32)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                return mask

        path = os.path.join(mask_folder, "mask.png")
        if os.path.isfile(path):
            mask = cv2.imread(
                path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = (mask > 0).astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            return mask

    # Fallback: extract from alpha channel
    if views:
        return extract_alpha_mask(views)

    return None


# ---------------------------------------------------------------------------
# Data preprocessing (reproduces realdata.py logic faithfully)
# ---------------------------------------------------------------------------

def preprocess_for_pose(images, mask, max_image_res=4096, mask_margin=8):
    """Crop, resize and normalize images exactly as realdata.py does.

    Args:
        images: list of numpy arrays (H, W, 3), same dtype (uint8/uint16)
        mask: (H, W) float32 array with 0/1, or None (no mask)
        max_image_res: maximum resolution (multiple of 512)
        mask_margin: pixel margin around mask bounding box

    Returns:
        I: (H_resized, W_resized, 3, N) float32 normalized array
        mask_out: (H_resized, W_resized, 1) float32 array
        roi: [h0, w0, r_s, r_e, c_s, c_e] for uncropping
    """
    h0 = images[0].shape[0]
    w0 = images[0].shape[1]
    margin = mask_margin

    mask_flag = False
    if mask is not None:
        mask_flag = True

    if not mask_flag:
        mask = np.ones((h0, w0), np.float32)

    # Resize mask to match image dimensions if needed
    if mask.shape[0] != h0 or mask.shape[1] != w0:
        logger.info("Resizing mask from %dx%d to %dx%d to match images",
                     mask.shape[0], mask.shape[1], h0, w0)
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

    # Compute crop bounding box from mask (same logic as realdata.py)
    rows, cols = np.nonzero(mask)
    rowmin = np.min(rows)
    rowmax = np.max(rows)
    row = rowmax - rowmin
    colmin = np.min(cols)
    colmax = np.max(cols)
    col = colmax - colmin

    if mask_flag:
        # Check if margin fits
        if (rowmin - margin <= 0 or rowmax + margin > h0 or
                colmin - margin <= 0 or colmax + margin > w0):
            flag = False
        else:
            flag = True

        if row > col and flag:
            r_s = rowmin - margin
            r_e = rowmax + margin
            c_s = max(colmin - int(0.5 * (row - col)) - margin, 0)
            c_e = min(colmax + int(0.5 * (row - col)) + margin, w0)
        elif col >= row and flag:
            r_s = max(rowmin - int(0.5 * (col - row)) - margin, 0)
            r_e = min(rowmax + int(0.5 * (col - row)) + margin, h0)
            c_s = colmin - margin
            c_e = colmax + margin
        else:
            # fallback: no margin fits
            r_s = rowmin - margin if rowmin - margin > 0 else 0
            r_e = rowmax + margin if rowmax + margin <= h0 else h0
            c_s = colmin - margin if colmin - margin > 0 else 0
            c_e = colmax + margin if colmax + margin <= w0 else w0

        if not flag:
            r_s = 0
            r_e = h0
            c_s = 0
            c_e = w0
    else:
        # No mask: make square crop from center
        margin = 0
        flag = True
        if row <= col and flag:
            r_s = rowmin - margin
            r_e = rowmax + margin
            c_s = int(0.5 * col) - int(0.5 * row)
            c_e = int(0.5 * col) + int(0.5 * row)
        elif row > col and flag:
            r_s = int(0.5 * row) - int(0.5 * col)
            r_e = int(0.5 * row) + int(0.5 * col)
            c_s = colmin - margin
            c_e = colmax + margin

    # Crop mask
    mask = mask[r_s:r_e, c_s:c_e]

    # Determine resize resolution (multiple of 512, capped at max_image_res)
    crop_h = r_e - r_s
    crop_w = c_e - c_s
    h = int(np.floor(max(crop_h, crop_w) / 512) * 512)
    if h > max_image_res:
        h = max_image_res
    if h < 512:
        h = 512
    w = h

    logger.info("Original size: %d x %d, crop: %d x %d, resize: %d x %d",
                h0, w0, crop_h, crop_w, h, w)

    # Resize mask
    mask = np.float32(
        cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_CUBIC) > 0.5)

    n_images = len(images)
    I = np.zeros((n_images, h, w, 3), np.float32)

    for i, img in enumerate(images):
        # Crop
        if flag or mask_flag:
            img = img[r_s:r_e, c_s:c_e, :]

        # Resize
        img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)

        # Convert to float
        if img.dtype == np.uint8:
            bit_depth = 255.0
        elif img.dtype == np.uint16:
            bit_depth = 65535.0
        else:
            bit_depth = 1.0
        img = np.float32(img) / bit_depth

        I[i, :, :, :] = img

    # Data normalization (same as realdata.py)
    I_flat = np.reshape(I, (n_images, h * w, 3))
    mask_flat = mask.flatten()
    masked_pixels = np.sum(mask_flat == 1)
    if masked_pixels > 0:
        temp = np.mean(I_flat[:, mask_flat == 1, :], axis=2)  # (N, num_masked)
        mx = np.max(temp, axis=1)  # (N,)
    else:
        logger.warning("Empty mask after crop/resize, skipping normalization")
        mx = np.ones(n_images)
    I_flat /= (mx.reshape(-1, 1, 1) + 1.0e-6)

    # Reshape to (h, w, 3, N) as expected by the model
    I_flat = np.transpose(I_flat, (1, 2, 0))  # (h*w, 3, N)
    I_out = I_flat.reshape(h, w, 3, n_images)

    mask_out = mask.reshape(h, w, 1).astype(np.float32)
    roi = np.array([h0, w0, r_s, r_e, c_s, c_e])

    return I_out, mask_out, roi


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, target, pixel_samples, device):
    """Load SDM-UniPS models (normal and/or brdf) from checkpoint.

    Returns:
        (net_nml or None, net_brdf or None)
    """
    from sdm_unips.modules.model import model
    from sdm_unips.modules.model.model_utils import loadmodel

    net_nml = None
    net_brdf = None

    if "normal" in target:
        model_dir = f"{checkpoint_path}/normal"
        net_nml = model.Net(pixel_samples, "normal", device).to(device)
        net_nml = torch.nn.DataParallel(net_nml)
        pytmodel = "".join(glob.glob(f"{model_dir}/*.pytmodel"))
        net_nml = loadmodel(net_nml, pytmodel, strict=False)
        net_nml.module.no_grad()
        logger.info("Loaded normal model from %s", model_dir)

    if "brdf" in target:
        model_dir = f"{checkpoint_path}/brdf"
        net_brdf = model.Net(pixel_samples, "brdf", device).to(device)
        net_brdf = torch.nn.DataParallel(net_brdf)
        pytmodel = "".join(glob.glob(f"{model_dir}/*.pytmodel"))
        net_brdf = loadmodel(net_brdf, pytmodel, strict=False)
        net_brdf.module.no_grad()
        logger.info("Loaded brdf model from %s", model_dir)

    return net_nml, net_brdf


# ---------------------------------------------------------------------------
# Inference for a single pose
# ---------------------------------------------------------------------------

def infer_pose(I, mask, roi, net_nml, net_brdf, target, device,
               canonical_resolution=256, pixel_samples=10000, scalable=False):
    """Run SDM-UniPS inference on preprocessed data for one pose.

    Args:
        I: (H, W, 3, N) float32 array
        mask: (H, W, 1) float32 array
        roi: [h0, w0, r_s, r_e, c_s, c_e]
        net_nml: normal network (or None)
        net_brdf: brdf network (or None)
        target: 'normal', 'brdf', or 'normal_and_brdf'
        device: torch device
        canonical_resolution: int
        pixel_samples: int
        scalable: use scalable (tiled) inference

    Returns:
        dict with 'normal', 'albedo', 'roughness', 'metallic' keys
        (each full-resolution numpy array or None)
    """
    from sdm_unips.modules.model import decompose_tensors

    H, W = I.shape[0], I.shape[1]
    n_img = I.shape[3]
    h0, w0 = int(roi[0]), int(roi[1])
    r_s, r_e, c_s, c_e = int(roi[2]), int(roi[3]), int(roi[4]), int(roi[5])

    # Build tensors: I -> (B, C, H, W, N), mask -> (B, 1, H, W)
    I_t = torch.from_numpy(
        I.transpose(2, 0, 1, 3)  # (C, H, W, N)
    ).unsqueeze(0).to(device)  # (1, C, H, W, N)
    M_t = torch.from_numpy(
        mask.transpose(2, 0, 1)  # (1, H, W)
    ).unsqueeze(0).to(device)  # (1, 1, H, W)

    B = 1
    C = 3
    nImgArray = torch.tensor([n_img], dtype=torch.long)

    results = {}

    with torch.no_grad():
        if scalable:
            patch_size = 512
            patches_I = decompose_tensors.divide_tensor_spatial(
                I_t.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W),
                block_size=patch_size, method="tile_stride")
            patches_I = patches_I.reshape(
                B, n_img, -1, C, patch_size, patch_size
            ).permute(0, 2, 3, 4, 5, 1)
            sliding_blocks = patches_I.shape[1]
            patches_M = decompose_tensors.divide_tensor_spatial(
                M_t, block_size=patch_size, method="tile_stride")

            patches_nml = []
            patches_base = []
            patches_rough = []
            patches_metal = []

            for k in range(sliding_blocks):
                logger.info("Recovering %s map(s): %d / %d",
                            target, k + 1, sliding_blocks)
                if torch.sum(patches_M[:, k, :, :, :]) > 0:
                    pI = patches_I[:, k, :, :, :, :]
                    pI = F.interpolate(
                        pI.permute(0, 4, 1, 2, 3).reshape(
                            -1, pI.shape[1], pI.shape[2], pI.shape[3]),
                        size=(patch_size, patch_size), mode="bilinear",
                        align_corners=True,
                    ).reshape(B, n_img, C, patch_size, patch_size).permute(
                        0, 2, 3, 4, 1)
                    pM = F.interpolate(
                        patches_M[:, k, :, :, :],
                        size=(patch_size, patch_size), mode="bilinear",
                        align_corners=True)
                    nout = torch.zeros((B, 3, patch_size, patch_size))
                    bout = torch.zeros((B, 3, patch_size, patch_size))
                    rout = torch.zeros((B, 1, patch_size, patch_size))
                    mout = torch.zeros((B, 1, patch_size, patch_size))
                    if "normal" in target and net_nml is not None:
                        nout, _, _, _ = net_nml(
                            pI, pM, nImgArray.reshape(-1, 1),
                            decoder_resolution=patch_size * torch.ones(
                                pI.shape[0], 1),
                            canonical_resolution=canonical_resolution
                            * torch.ones(pI.shape[0], 1))
                        nout = (F.interpolate(
                            nout, size=(patch_size, patch_size),
                            mode="bilinear", align_corners=True) * pM).cpu()
                    if "brdf" in target and net_brdf is not None:
                        _, bout, rout, mout = net_brdf(
                            pI, pM, nImgArray.reshape(-1, 1),
                            decoder_resolution=patch_size * torch.ones(
                                pI.shape[0], 1),
                            canonical_resolution=canonical_resolution
                            * torch.ones(pI.shape[0], 1))
                        bout = F.interpolate(
                            bout, size=(patch_size, patch_size),
                            mode="bilinear", align_corners=True).cpu()
                        rout = F.interpolate(
                            rout, size=(patch_size, patch_size),
                            mode="bilinear", align_corners=True).cpu()
                        mout = F.interpolate(
                            mout, size=(patch_size, patch_size),
                            mode="bilinear", align_corners=True).cpu()
                    patches_nml.append(nout)
                    patches_base.append(bout)
                    patches_rough.append(rout)
                    patches_metal.append(mout)
                else:
                    patches_nml.append(
                        torch.zeros((B, 3, patch_size, patch_size)))
                    patches_base.append(
                        torch.zeros((B, 3, patch_size, patch_size)))
                    patches_rough.append(
                        torch.zeros((B, 1, patch_size, patch_size)))
                    patches_metal.append(
                        torch.zeros((B, 1, patch_size, patch_size)))

            patches_nml = torch.stack(patches_nml, dim=1)
            patches_base = torch.stack(patches_base, dim=1)
            patches_rough = torch.stack(patches_rough, dim=1)
            patches_metal = torch.stack(patches_metal, dim=1)
            nml = decompose_tensors.merge_tensor_spatial(
                patches_nml.permute(1, 0, 2, 3, 4),
                method="tile_stride").squeeze().permute(1, 2, 0)
            base = decompose_tensors.merge_tensor_spatial(
                patches_base.permute(1, 0, 2, 3, 4),
                method="tile_stride").squeeze().permute(1, 2, 0)
            rough = decompose_tensors.merge_tensor_spatial(
                patches_rough.permute(1, 0, 2, 3, 4),
                method="tile_stride").squeeze()
            metal = decompose_tensors.merge_tensor_spatial(
                patches_metal.permute(1, 0, 2, 3, 4),
                method="tile_stride").squeeze()
        else:
            logger.info("Recovering %s map(s) 1 / 1", target)
            nml = torch.zeros((H, W, 3))
            base = torch.zeros((H, W, 3))
            rough = torch.zeros((H, W))
            metal = torch.zeros((H, W))

            if "normal" in target and net_nml is not None:
                nout, _, _, _ = net_nml(
                    I_t, M_t, nImgArray.reshape(-1, 1),
                    decoder_resolution=H * torch.ones(I_t.shape[0], 1),
                    canonical_resolution=canonical_resolution
                    * torch.ones(I_t.shape[0], 1))
                nml = (nout * M_t).squeeze().permute(1, 2, 0).cpu().detach()
                del nout

            if "brdf" in target and net_brdf is not None:
                _, bout, rout, mout_t = net_brdf(
                    I_t, M_t, nImgArray.reshape(-1, 1),
                    decoder_resolution=H * torch.ones(I_t.shape[0], 1),
                    canonical_resolution=canonical_resolution
                    * torch.ones(I_t.shape[0], 1))
                base = (bout * M_t).squeeze().permute(
                    1, 2, 0).cpu().detach()
                rough = (rout * M_t).squeeze().cpu().detach()
                metal = (mout_t * M_t).squeeze().cpu().detach()
                del bout, rout, mout_t

    # Uncrop results back to original resolution (same as builder.py)
    if "normal" in target and net_nml is not None:
        nml_np = nml.cpu().numpy()
        nml_np = cv2.resize(
            nml_np, dsize=(c_e - c_s, r_e - r_s),
            interpolation=cv2.INTER_CUBIC)
        nml_mask = np.float32(
            np.abs(1 - np.sqrt(np.sum(nml_np * nml_np, axis=2))) < 0.5)
        nml_np = np.divide(
            nml_np, np.linalg.norm(nml_np, axis=2, keepdims=True) + 1.0e-12)
        nml_np = nml_np * nml_mask[:, :, np.newaxis]
        nout_full = np.zeros((h0, w0, 3), np.float32)
        nout_full[r_s:r_e, c_s:c_e, :] = nml_np
        results["normal"] = nout_full
    else:
        results["normal"] = None

    if "brdf" in target and net_brdf is not None:
        base_np = base.cpu().numpy() if isinstance(base, torch.Tensor) \
            else base
        rough_np = rough.cpu().numpy() if isinstance(rough, torch.Tensor) \
            else rough
        metal_np = metal.cpu().numpy() if isinstance(metal, torch.Tensor) \
            else metal

        base_np = cv2.resize(
            base_np, dsize=(c_e - c_s, r_e - r_s),
            interpolation=cv2.INTER_CUBIC)
        rough_np = cv2.resize(
            rough_np, dsize=(c_e - c_s, r_e - r_s),
            interpolation=cv2.INTER_CUBIC)
        metal_np = cv2.resize(
            metal_np, dsize=(c_e - c_s, r_e - r_s),
            interpolation=cv2.INTER_CUBIC)

        bout_full = np.zeros((h0, w0, 3), np.float32)
        bout_full[r_s:r_e, c_s:c_e, :] = base_np
        results["albedo"] = bout_full

        rout_full = np.zeros((h0, w0), np.float32)
        rout_full[r_s:r_e, c_s:c_e] = rough_np
        results["roughness"] = rout_full

        mout_full = np.zeros((h0, w0), np.float32)
        mout_full[r_s:r_e, c_s:c_e] = metal_np
        results["metallic"] = mout_full
    else:
        results["albedo"] = None
        results["roughness"] = None
        results["metallic"] = None

    return results


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_normal_16bit(normal, out_path):
    """Save normal map as 16-bit PNG (BGR convention for cv2)."""
    # normal is (H, W, 3) in [-1, 1], convert to [0, 65535]
    normal_16 = np.uint16(
        np.clip(65535 * (0.5 * (1 + normal[:, :, ::-1])), 0, 65535))
    cv2.imwrite(out_path, normal_16, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_color_16bit(color, out_path):
    """Save a color map (albedo) as 16-bit PNG."""
    color_16 = np.uint16(np.clip(65535 * color[:, :, ::-1], 0, 65535))
    cv2.imwrite(out_path, color_16, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_gray_16bit(gray, out_path):
    """Save a single-channel map (roughness/metallic) as 16-bit PNG."""
    gray_16 = np.uint16(np.clip(65535 * gray, 0, 65535))
    cv2.imwrite(out_path, gray_16, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_sfm_inference(sfm_path, output_folder, checkpoint_path,
                      mask_folder=None, mask_output_folder=None,
                      nb_img=10, downscale=1,
                      use_cuda=True, target="normal_and_brdf",
                      max_image_res=4096, canonical_resolution=256,
                      pixel_samples=10000, scalable=False,
                      mask_margin=8):
    """Run SDM-UniPS inference on all poses in an SfM file.

    Args:
        sfm_path: path to input SfMData JSON or .sfm file
        output_folder: where to write output maps and JSON
        checkpoint_path: path to checkpoint directory (with normal/ and brdf/)
        mask_folder: optional folder with mask PNGs (named by poseId/viewId)
        nb_img: number of images per pose (10 by default)
        downscale: integer downscale factor
        use_cuda: use GPU
        target: 'normal', 'brdf', or 'normal_and_brdf'
        max_image_res: maximum image resolution (multiple of 512)
        canonical_resolution: canonical resolution for the model
        pixel_samples: number of pixel samples for the model
        scalable: use scalable (tiled) processing for large images
        mask_margin: pixel margin around mask bounding box

    Returns:
        Path to the output JSON file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load SfM data
    sfm_data = load_sfm(sfm_path)
    pose_groups = group_views_by_pose(sfm_data)
    logger.info("Loaded %d views, %d poses",
                len(sfm_data.get("views", [])), len(pose_groups))

    # Load model
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available()
                          else "cpu")
    logger.info("Loading SDM-UniPS model on %s...", device)
    net_nml, net_brdf = load_model(
        checkpoint_path, target, pixel_samples, device)
    logger.info("Model loaded (target=%s, canonical_resolution=%d, "
                "pixel_samples=%d, scalable=%s)",
                target, canonical_resolution, pixel_samples, scalable)

    # Process each pose
    results = []
    total_start = time.time()

    for pose_id, views in pose_groups.items():
        logger.info("=== Pose %s (%d views) ===", pose_id, len(views))
        pose_start = time.time()

        try:
            # Load images
            images = load_images_for_pose(views, nb_img, downscale)

            # Load mask
            view_ids = [str(v["viewId"]) for v in views]
            mask = find_mask_for_pose(pose_id, mask_folder, view_ids, views=views)

            # Save extracted mask at output resolution (match downscaled images)
            if mask is not None and mask_output_folder and not mask_folder:
                os.makedirs(mask_output_folder, exist_ok=True)
                mask_path = os.path.join(mask_output_folder, f"{pose_id}.png")
                mask_to_save = mask
                # Resize mask to match downscaled image dimensions
                img_h, img_w = images[0].shape[:2]
                if mask.shape[0] != img_h or mask.shape[1] != img_w:
                    mask_to_save = cv2.resize(
                        mask, (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(mask_path, np.uint8(mask_to_save * 255))
                logger.info("Saved mask to %s", mask_path)

            # Preprocess (crop, resize, normalize) -- reproduces realdata.py
            I, mask_out, roi = preprocess_for_pose(
                images, mask,
                max_image_res=max_image_res,
                mask_margin=mask_margin,
            )

            # Run inference
            outputs = infer_pose(
                I, mask_out, roi,
                net_nml, net_brdf, target, device,
                canonical_resolution=canonical_resolution,
                pixel_samples=pixel_samples,
                scalable=scalable,
            )

            # Save outputs
            pose_results = {
                "poseId": pose_id,
                "viewId": str(views[0].get("viewId")),
                "nbImages": len(images),
            }

            if outputs["normal"] is not None:
                nml_path = os.path.join(
                    output_folder, f"{pose_id}_normals.png")
                save_normal_16bit(outputs["normal"], nml_path)
                pose_results["normalMapPath"] = os.path.abspath(nml_path)
                pose_results["width"] = outputs["normal"].shape[1]
                pose_results["height"] = outputs["normal"].shape[0]
                logger.info("Saved normal map: %s", nml_path)

            if outputs["albedo"] is not None:
                albedo_path = os.path.join(
                    output_folder, f"{pose_id}_albedo.png")
                save_color_16bit(outputs["albedo"], albedo_path)
                pose_results["albedoMapPath"] = os.path.abspath(albedo_path)
                logger.info("Saved albedo map: %s", albedo_path)

            if outputs["roughness"] is not None:
                rough_path = os.path.join(
                    output_folder, f"{pose_id}_roughness.png")
                save_gray_16bit(outputs["roughness"], rough_path)
                pose_results["roughnessMapPath"] = os.path.abspath(rough_path)
                logger.info("Saved roughness map: %s", rough_path)

            if outputs["metallic"] is not None:
                metal_path = os.path.join(
                    output_folder, f"{pose_id}_metallic.png")
                save_gray_16bit(outputs["metallic"], metal_path)
                pose_results["metallicMapPath"] = os.path.abspath(metal_path)
                logger.info("Saved metallic map: %s", metal_path)

            results.append(pose_results)

            pose_time = time.time() - pose_start
            logger.info("Pose %s done in %.1fs", pose_id, pose_time)

        except Exception as e:
            logger.error("Failed on pose %s: %s", pose_id, e, exc_info=True)
            continue

    total_time = time.time() - total_start
    logger.info("All poses processed in %.1fs", total_time)

    logger.info("Inference complete: %d poses processed", len(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SDM-UniPS inference from SfM JSON")
    parser.add_argument("--input", "-i", required=True,
                        help="Input SfMData JSON or .sfm file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output folder for maps and JSON")
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to checkpoint directory "
                             "(containing normal/ and brdf/ subdirs)")
    parser.add_argument("--masks", "-m", default=None,
                        help="Folder with mask PNGs "
                             "(named by poseId/viewId)")
    parser.add_argument("--nb-img", type=int, default=10,
                        help="Number of images per pose (default: 10)")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Integer downscale factor (1 = no downscale)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use GPU")
    parser.add_argument("--target", default="normal_and_brdf",
                        choices=["normal", "brdf", "normal_and_brdf"],
                        help="What to estimate (default: normal_and_brdf)")
    parser.add_argument("--max-image-res", type=int, default=4096,
                        help="Maximum image resolution (default: 4096)")
    parser.add_argument("--canonical-resolution", type=int, default=256,
                        help="Canonical resolution (default: 256)")
    parser.add_argument("--pixel-samples", type=int, default=10000,
                        help="Number of pixel samples (default: 10000)")
    parser.add_argument("--scalable", action="store_true",
                        help="Use scalable tiled inference")
    parser.add_argument("--mask-margin", type=int, default=8,
                        help="Pixel margin around mask bbox (default: 8)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_sfm_inference(
        sfm_path=args.input,
        output_folder=args.output,
        checkpoint_path=args.checkpoint,
        mask_folder=args.masks,
        nb_img=args.nb_img,
        downscale=args.downscale,
        use_cuda=args.cuda,
        target=args.target,
        max_image_res=args.max_image_res,
        canonical_resolution=args.canonical_resolution,
        pixel_samples=args.pixel_samples,
        scalable=args.scalable,
        mask_margin=args.mask_margin,
    )


if __name__ == "__main__":
    main()
