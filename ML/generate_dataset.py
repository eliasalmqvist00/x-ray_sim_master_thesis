import os
import sys
from pathlib import Path
import numpy as np
from gvxrPython3 import gvxr
from gvxrPython3.JSON2gVXRDataReader import *
import json
import tifffile as tiff
from skimage.filters import threshold_otsu

xsim_root = Path(__file__).resolve().parent.parent
sys.path.append(str(xsim_root))
meshes_root = xsim_root / "Data/Cotton fibre/Meshes/"

#own class imports
from ML.tomograph import Tomograph

# DEFINE SCANNING PARAMETERS 
"--geometry-- "
SDD = 20.42 
zoom = 1
SSD = 11 + zoom
xPixels = 996            
yPixels = 996            
pixel_size = 0.00135      #effective pixel size

"--source--"
voltage_kV = 60         
current_mA = 1  
exposure_time_s = 4
filter_thickness = 0.8

"--detector"
epsilon = 0.01        # 0.01 - 0.02 (defines amount of blur)

"--sample--"            
density = 1.3
offset = (0,0,0)         
rotation =(0, 10, 0)

nbr_projections = 1300
reconstruction_filter="hann"
FFC = 50

# DEFINE OUTPUT PATH 
output_path = xsim_root / f"ML/Datasets/sample_E0004"
os.makedirs(output_path, exist_ok=False)

# DEFINE MESH PATH 
meshes_path = xsim_root / "Data/Cotton fibre/Meshes/cotton_fibres_scaled_moved"

def save_metadata(output_path, **params):
    with open(output_path / "metadata.json", "w") as f:
        json.dump(params, f, indent=2)

def get_meshes(path):
    folder = Path(str(path))      
    files = folder.glob(f"{"fibre"}_*.stl")
    meshes = [str(f) for f in files]
    return meshes

def add_fibers(xct, meshes, density, offset, rotation):
    cotton_mix = ["C", "H", "O"]
    cotton_wf = [0.444, 0.062, 0.494]
    
    i = 1
    for mesh in meshes:
        xct.addMesh(f"fibre_{i}", mesh, density=density, mixture=cotton_mix, mass_fraction=cotton_wf, unit="mm")

        dx, dy, dz = rotation
        gvxr.rotateNode(
                f"fibre_{i}", dx,dy,dz)
            
        gx, gy, gz = offset
        if gx != 0 or gy != 0 or gz != 0:
            gvxr.translateNode(f"fibre_{i}", gx, gy, gz, "mm")
        i += 1


def segmentation(xct, nbr_projections):
        recon = xct.acquisition_reconstruction(path=output_path, num_of_projs=nbr_projections, 
                                               filter = reconstruction_filter)
        
        mask = recon >= threshold_otsu(recon)
        
        segmented = np.zeros_like(recon, dtype=np.uint8)
        segmented[mask] = 1
        tiff.imwrite(str(output_path / "mask.tif"), segmented.astype(np.float32))

def generate_mask():
    tom_noise_free = Tomograph()
    tom_noise_free.set_geometry(SSD=SSD, SDD=SDD, xPixels=xPixels, yPixels=yPixels, pixel_size=pixel_size)
    tom_noise_free.setup_mono_spectrum(voltage_kV=voltage_kV)
    tom_noise_free.setup_detector()

    meshes = get_meshes(meshes_path)
    add_fibers(tom_noise_free, meshes, density, offset, rotation)
    segmentation(tom_noise_free, nbr_projections)

def generate_noisy():
    tom_noisy = Tomograph()
    tom_noisy.set_geometry(SSD=SSD, SDD=SDD, xPixels=xPixels, yPixels=yPixels, pixel_size=pixel_size)
    tom_noisy.setup_poly_spectrum(voltage_kV=voltage_kV, current_mA=current_mA, 
                                  exposure_time_s=exposure_time_s, filter_tickness=filter_thickness)
    tom_noisy.setup_detector(shot_noise=True, epsilon=epsilon)

    meshes = get_meshes(meshes_path)
    add_fibers(tom_noisy, meshes, density, offset, rotation)
    recon = tom_noisy.acquisition_reconstruction(path=output_path, num_of_projs=nbr_projections, 
                                                 filter = reconstruction_filter, FFC=FFC, save_recon=True, label="noisy")
    
if __name__ == "__main__":
    generate_mask()
    generate_noisy()

    save_metadata(
    output_path,
    SDD = SDD,
    SSD = SSD,
    xPixels = xPixels,         
    yPixels = yPixels,            
    pixel_size = pixel_size,  
    mAs = current_mA*exposure_time_s,
    voltage_kV = voltage_kV,           
    filter_thickness = filter_thickness,              
    epsilon = epsilon,             
    density = density,
    offset = offset,       
    rotation = rotation,       
    nbr_projections = nbr_projections,
    reconstruction_filter=reconstruction_filter,
    FFC=FFC
)
