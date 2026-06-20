from gvxrPython3 import gvxr
from gvxrPython3.gVXRDataReader import *
from gvxrPython3.utils import loadSpekpySpectrum
from gvxrPython3 import gvxr2json
from gvxrPython3.JSON2gVXRDataReader import *

import time

import tifffile as tiff
import os
import numpy as np
import cil as cil
from cil.io import TIFFWriter
from cil.processors import TransmissionAbsorptionConverter
from cil.recon import FDK
import shutil


class Tomograph:
    def __init__(self, length_unit = "mm"):
        self.lu = length_unit
        gvxr.createOpenGLContext()


    def set_geometry(self, SSD, SDD, xPixels ,yPixels, pixel_size):
        gvxr.setSourcePosition(0, SSD, 0, self.lu)
        gvxr.setDetectorPosition(0, -(SDD-SSD), 0, self.lu)

        gvxr.usePointSource()

        gvxr.setDetectorUpVector(0,0,-1)
        gvxr.setDetectorNumberOfPixels(xPixels, yPixels)
        gvxr.setDetectorPixelSize(pixel_size, pixel_size, self.lu)

    def setup_poly_spectrum(self, voltage_kV, current_mA, exposure_time_s, th_deg=12, filter_tickness = 0.26): 
        SDD_cm = np.linalg.norm(np.array(gvxr.getSourcePosition("cm")) - np.array(gvxr.getDetectorPosition("cm")))
        mAs = exposure_time_s*current_mA

        loadSpekpySpectrum(voltage_kV, filters = [["Al", filter_tickness]], 
                                    th_in_deg=th_deg, max_number_of_energy_bins=50,
                                    mAs=mAs, z=SDD_cm)   
        

    def setup_mono_spectrum(self, voltage_kV):
        gvxr.setMonoChromatic(voltage_kV, "keV", 2000)

    def setup_detector(self, scintillator=('CdWO4', 525), shot_noise=False, epsilon = 0):
        if scintillator != None:
            scint_mat, scint_thick = scintillator
            gvxr.setScintillator(scint_mat, scint_thick, "um")
        
        if shot_noise:  
            gvxr.enablePoissonNoise()
        else:
            gvxr.disablePoissonNoise()

        if epsilon > 0:
            x = np.linspace(-2,2,41)
            coeff1 = 1.0/np.sqrt(2.0*(np.pi)*np.absolute(epsilon))
            coeff2 = -(np.power(x, 2)/(2.0*epsilon))
            lsf = coeff1*np.exp(coeff2)
            lsf /= np.sum(lsf)
            gvxr.setLSF(lsf)
        else: 
            gvxr.clearLSF()


    def addMesh(self, label, path, density, mixture = None, mass_fraction = None, unit = "mm"): 
        gvxr.loadMeshFile(label, str(path), unit)
        gvxr.setMixture(label, mixture, mass_fraction)
        gvxr.setDensity(label, density, "g/cm3")


    def viewScene(self):
        gvxr.displayScene()
        gvxr.renderLoop()


    def compute2D(self):
        return np.array(gvxr.computeXRayImage(), dtype=np.single) / gvxr.getTotalEnergyWithDetectorResponse();

    
    def acquisition_reconstruction(self, path, num_of_projs, filter, FFC= 0, save_projections = False, save_recon=False, label = ""):
        ct_path = os.path.join(path, "Projections/")
        start_acq = time.time()
        gvxr.computeCTAcquisition(ct_path,
                                    "",
                                    num_of_projs,
                                    0,
                                    False,
                                    360,
                                    FFC, 
                                    *[0,0,0],
                                    "mm", 
                                    *(0,0,-1),
                                    True
        )
        
        end_acq = time.time()
        print("CT-Acquisition complete! Execution time: "+ str(np.round(end_acq-start_acq, 4)) + " s")

        json_fname = os.path.join(path, "simulation-" + str(num_of_projs) + ".json")
        gvxr2json.saveJSON(json_fname)

        reader = JSON2gVXRDataReader(json_fname)
        data = reader.read()
        data = TransmissionAbsorptionConverter(white_level=data.max())(data)
        data.reorder(order='tigre') 
        data_image_geometry = data.geometry.get_ImageGeometry()

        start_recon = time.time()
        recon = FDK(data, data_image_geometry, filter=filter).run()
        end_recon = time.time()
        print("FDK-reconstruction complete! Execution time: " + str(np.round(end_recon-start_recon,4)) + " s")

        recon_np = recon.as_array()

        if save_recon:
            stack_path = os.path.join(path, f"{label}.tif")
            tiff.imwrite(stack_path, recon_np)

        if save_projections is False:
            shutil.rmtree(ct_path)
            os.remove(json_fname)            
        return recon_np

    

