import numpy as np
from pathlib import Path
import sys
import os
import tifffile as tiff
import json

xsim_root = Path(__file__).resolve().parent.parent
sys.path.append(str(xsim_root))

test_path = xsim_root / "Data/Cotton fibre/Testset"
train_val_path = xsim_root / "ML/Datasets/"

output_path = xsim_root / f"ML/Processed/set_E0005"

comment = "Elias samples E01-E06 without edges"

CROP_EDGE = 180
STEP = 10                     

train_ids = {1, 2, 3, 4}
val_ids = {5,6}

phase_edge_ids = {}

sample_type = "E"

normalisation = "min max"


def save_metadata(output_path, **params):
    with open(output_path / "metadata.json", "w") as f:
        json.dump(params, f, indent=2)

def get_sample_id(sample_dir):
    return int(sample_dir.name.split(f"_{sample_type}")[1])

def normalise_volume(slice_arr: np.ndarray) -> np.ndarray:
    arr = slice_arr.astype(np.float32)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr

def remove_outliers(vol, bottom_percentile, top_percentile):
    mask = vol > 0
    p_bottom = np.percentile(vol[mask], bottom_percentile)
    p_top = np.percentile(vol[mask], top_percentile)

    return np.clip(vol, p_bottom, p_top)

def add_phase_edges(sim_vol, mask):
    from scipy.ndimage import binary_erosion, binary_dilation
    from scipy.ndimage import gaussian_filter

    def add_inside_outside_edges(mask, air, fibre, outside_width, inside_width, outside_ratio=0.01, inside_ratio=1.7):
        """
        mask: binary volume (0 background, 1 object)
        air: background(air) value
        fibre: fibre (slice.max) value
        outside_width: thickness of edges outside the fibres
        inside_width: thickness of edges inside the fibres
        outside_ratio: ratio of outside edge vs air (taken from experimental data)
        inside_ratio: ratio of inside edge vs fibre (taken from experimental data)
        """
        mask = mask.astype(bool)
        out = mask.astype(np.float32)

        # ---------- OUTSIDE EDGE ----------
        dilated = binary_dilation(mask, iterations=outside_width)
        outside_edge = dilated & (~mask)

        # ---------- INSIDE EDGE ----------
        eroded = binary_erosion(mask, iterations=inside_width)
        inside_edge = mask & (~eroded)

        if air < 0:
            outside_val = -1*np.abs(air)*(1/outside_ratio)
        else:
            outside_val = air*outside_ratio

        inside_val = fibre*inside_ratio

        # Assign values
        out[outside_edge] = outside_val
        out[inside_edge] = inside_val

        return out

    def add_phase_edges(sim_vol, mask):

        Z = sim_vol.shape[0]
        
        sim_vol_with_phase = []

        for z in range(Z):
            slice = sim_vol[z]
            mask_slice = mask[z]

            air = slice.min()
            fibre = slice[mask_slice == 1].mean()
            phase_edge = add_inside_outside_edges(mask_slice, air, fibre, 3, 2, outside_ratio=0.3, inside_ratio=4.3)
            
            phase_only = np.where((phase_edge != 0) & (phase_edge != 1),
                                phase_edge - slice,
                                0)

            phase_blur = gaussian_filter(phase_only, sigma=0.6)
            slice_with_phase = slice + phase_blur

            sim_vol_with_phase.append(slice_with_phase)
        
        sim_vol_with_phase = np.array(sim_vol_with_phase)

        return sim_vol_with_phase
    
    return add_phase_edges(sim_vol=sim_vol, mask=mask)


def generate_dataset(root_dir, split_ids):
    noisy_list, mask_list = [], []

    for sample_dir in sorted(root_dir.glob(f"sample_{sample_type}0*")):
        sample_id = get_sample_id(sample_dir)

        if sample_id not in split_ids:
            continue

        noisy = tiff.imread(sample_dir / "noisy.tif")  
        mask = tiff.imread(sample_dir / "mask.tif")
        
        noisy_sel = noisy[10::STEP, CROP_EDGE:-CROP_EDGE, CROP_EDGE:-CROP_EDGE]
        mask_sel  = mask [10::STEP, CROP_EDGE:-CROP_EDGE, CROP_EDGE:-CROP_EDGE]

        if sample_id in phase_edge_ids:
            noisy_sel = add_phase_edges(sim_vol=noisy_sel, mask=mask_sel)

        noisy_list.append(noisy_sel)
        mask_list.append(mask_sel)
    
    noisy_list = np.concatenate(noisy_list, axis=0)
    mask_list = np.concatenate(mask_list, axis=0)

    print("size:", noisy_list.shape, mask_list.shape)
    print("noisy min:", noisy.min(), "max:", noisy.max())
    noisy_list=normalise_volume(noisy_list)
    return noisy_list, mask_list

def generate_test(root_dir):
    noisy = tiff.imread(root_dir / "test_vol.tif")  
    mask = tiff.imread(root_dir / "test_gt.tif")

    noisy_sel = noisy[:, CROP_EDGE:-CROP_EDGE, CROP_EDGE:-CROP_EDGE]
    mask_sel  = mask [:, CROP_EDGE:-CROP_EDGE, CROP_EDGE:-CROP_EDGE]

    noisy_sel = remove_outliers(noisy_sel, 1, 99)
    noisy_sel = normalise_volume(noisy_sel)
    return noisy_sel, mask_sel

if __name__ == "__main__":

    test_dir = Path(test_path)
    train_val_dir = Path(train_val_path)
    output_dir = Path(output_path)
    output_path.mkdir(exist_ok=True)

    train_noisy, train_mask = generate_dataset(train_val_dir, train_ids)
    val_noisy, val_mask = generate_dataset(train_val_dir, val_ids)
    test_noisy, test_mask = generate_test(test_dir)


    tiff.imwrite(output_dir / "train_vol.tif", train_noisy)
    tiff.imwrite(output_dir / "train_mask.tif", train_mask)
    tiff.imwrite(output_dir / "val_vol.tif",   val_noisy)
    tiff.imwrite(output_dir / "val_mask.tif",    val_mask)
    tiff.imwrite(output_dir / "test_vol.tif",   test_noisy)
    tiff.imwrite(output_dir / "test_mask.tif",    test_mask)

    save_metadata(
    output_path,
    val_ids = list(val_ids),
    train_ids = list(train_ids),
    phase_edge_ids = list(phase_edge_ids),
    normlisation = normalisation,
    crop = CROP_EDGE,
    step = STEP,
    comment = comment
)


