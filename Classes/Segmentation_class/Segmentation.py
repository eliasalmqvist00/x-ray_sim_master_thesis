import numpy as np
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu
from pathlib import Path
import sys
import os
import shutil
import tifffile as tiff
from gvxrPython3 import gvxr
from gvxrPython3.gVXRDataReader import *
from gvxrPython3.JSON2gVXRDataReader import *

sys.path.append(Path("C:/Users/SEALMQVISTE/.vscode/Git-MastersThesis/x-ray_sim/Classes"))
from Classes.Nanotom_class.NanotomXCT import NanotomXCT

class Segmentation:

    def __init__(self, vtom:NanotomXCT):
        """
        Constructor of the Segmentation-class.

        :param vtom: The virtual CT-scanner used to aquire and reconstruct the CT-data for segmentation.
        """
        self.num_of_proj = np.round(vtom.detector_width)
        self.vtom = vtom


    def begin_segmentation(self, output_folder):
        """
        Begins the segmentation loop.

        :param output_folder: The folder in which the segmentation volume will be saved in.
        """
        self.output = output_folder
        self.segmentation_loop()


    def set_meshes_import(self, folder_path, prefix, mix:np.array):
        """
        Sets the folder from which the meshes to segment are imported from.

        :param folder_name: The folder path in which the meshes are located
        :param prefix: The prefix which each mesh name contains, for example "fibre"
        :param mix: The mixture and mass fraction, in for of a numpy array, that will be set for meshes in gvxr, ex. mix =(["C", "H"],[0.3, 0.7])
        """
        folder = Path(folder_path)
        
        files = folder.glob(f"{prefix}_*.stl")

        self.mix = mix
        self.prefix = prefix
        self.meshes = [str(f) for f in files]

        print(f"Number of meshes: {len(self.meshes)}")


    def segmentation_loop(self):
        """
        The loop that segments each fiber one at a time by performing a virtual CT scan 
        without noise and then uses otsu region grow to get the binary segmentation

        Saves the complete segmentation that is created after the loop is finished to
        output_folder which is set in begin_segmentation()
        """
        vtom = self.vtom

        self.vtom.setup_source(vtom.voltage, vtom.current, vtom.exp_time, noise=False, poly=False)
        segmented = np.zeros((int(vtom.detector_height), int(vtom.detector_width), int(vtom.detector_width)), dtype=np.uint16)

        label = 1
        for mesh in self.meshes:
            gvxr.removePolygonMeshesFromSceneGraph()
            vtom.addMesh(f"{self.prefix}_{label}", mesh, density=1.52, center=False, element="Au")

            ct_path = os.path.join(self.output, f"single_fibre_CT/fibre_{label}/")
            os.makedirs(ct_path, exist_ok=False)
            recon = self.acquisition_reconstruction(path=ct_path, num_of_projs=self.num_of_proj)

            mask = self.threshold_otsu(recon, tolerance=0.07)

            new_mask = (mask == 1) & (segmented == 0)

            segmented[new_mask] = label
            label += 1
            gvxr.removePolygonMeshesFromSceneGraph()

            shutil.rmtree(ct_path)

        tiff.imwrite(self.output + "/segmentation_labels.tif", segmented.astype(np.float32))


    def acquire_noise_data(self):
        vtom = self.vtom
        self.vtom.setup_source(vtom.voltage, vtom.current, vtom.exp_time, noise=True)
        i = 1
        for mesh in self.meshes:
            vtom.addMesh(f"{self.prefix}_{i}", mesh, density=1.52, center=False, mixture=self.mix[0], mass_fraction=self.mix[1])
            i += 1
        
        path = os.path.join(self.output, "CT_with_noise")
        os.makedirs(path, exist_ok=False)
        recon = self.acquisition_reconstruction(path, self.num_of_proj)
        tiff.imwrite(self.output + "/reconstruction_with_noise.tif", recon)


    def acquisition_reconstruction(self, path, num_of_projs):
        from gvxrPython3 import gvxr2json
        from cil.processors import TransmissionAbsorptionConverter
        from cil.recon import FDK

        ct_path = os.path.join(path, "Projections/")
        gvxr.computeCTAcquisition(ct_path,
                                    "",
                                    num_of_projs,
                                    0,
                                    False,
                                    360,
                                    0, 
                                    *[0,0,0],
                                    "mm", 
                                    *(0,0,-1),
                                    True
        )
        json_fname = os.path.join(path, "simulation-" + str(num_of_projs) + ".json")
        gvxr2json.saveJSON(json_fname)

        reader = JSON2gVXRDataReader(json_fname)
        data = reader.read()
        data = TransmissionAbsorptionConverter(white_level=data.max())(data)
        data.reorder(order='tigre') 
        data_image_geometry = data.geometry.get_ImageGeometry()

        recon = FDK(data, data_image_geometry, filter="shepp-logan").run()

        recon_np = recon.as_array()

        return recon_np


    def threshold_otsu(self, volume, tolerance=0.1):
        """
        Region-growing segmentation using Otsu threshold as seed.
        """

        # Compute Otsu threshold
        otsu = threshold_otsu(volume)

        # Seed: basic threshold
        seg = volume >= otsu

        return np.array(seg.astype(np.uint8))

