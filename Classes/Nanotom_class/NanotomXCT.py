from gvxrPython3 import gvxr
from gvxrPython3.gVXRDataReader import *
from gvxrPython3.utils import loadSpekpySpectrum
from gvxrPython3 import gvxr2json
from gvxrPython3.JSON2gVXRDataReader import *

from pathlib import Path

import time

import tifffile as tiff
import os
import numpy as np
import cil as cil
from cil.io import TIFFWriter
from cil.processors import TransmissionAbsorptionConverter
from cil.recon import FDK

class NanotomXCT:
    
    def __init__(self, scaling, output_path, length_unit="mm"):
        """
        Constructor.
        
        :param scaling: Binning of detector 
        :param output_path: The path to the folder in which the CT data will be saved in
        :param length_unit: Length units used
        """

        self.scaling = scaling
        self.lu = length_unit
        self.output_path = output_path
        gvxr.createOpenGLContext()
    
    def print_geometries(self):
        FoV = np.array([self.detector_width, self.detector_height]) * 0.1 / self.mag
        voxel_size = 0.1 / self.mag
        num_of_proj = int(np.round(self.detector_width * np.pi/2))

        print(f"FoV = ({FoV[0]}, {np.round(FoV[1],3)}) mm")
        print(f"Voxel size = {voxel_size*1000} um")
        print("Number of projections needed: " + str(num_of_proj))

    def set_distances(self, SSD, SDD):
        """
        Set distances between sample, detector and source 
        
        :param SSD: Source-Sample Distance
        :param SDD: Source-Detector Distance 
        """
        self.mag = SDD/SSD
        gvxr.setSourcePosition(0, SSD, 0, self.lu)
        gvxr.setDetectorPosition(0, -(SDD-SSD), 0, self.lu)


    def setup_source(self, voltage_kV, current_mA, exposure_time_s, th_deg=12, poly=True, filter=[["Al", 0.26]], noise=False):
        """
        Creates the X-ray source, eihter monochromatic or polychromatic. 

        :param voltage_kV: The tube voltage (in kV) of the X-ray source.
        :param current_mA: The X-ray tube current (in mA), the electron current flowing from the cathode (filament) to the anode (target)
        :param exposure_time_s: The exposure time (in seconds)
        :param th_deg: The anode (target) angle
        :param poly: Controls if the beam is polychromatic (True/False)
        :param filter: The filter (material, thickness (mm)) applied to the X-ray output.
        :param noise: Controls if the simulations are performed with noise (True/False)  
        """
        
        self.poly = poly
        gvxr.usePointSource()

        self.voltage = voltage_kV
        self.current = current_mA
        self.exp_time = exposure_time_s
        mAs = exposure_time_s*current_mA

        if poly:
            SDD_cm = np.linalg.norm(np.array(gvxr.getSourcePosition("cm")) - np.array(gvxr.getDetectorPosition("cm")))

            self.poly_spectrum = loadSpekpySpectrum(voltage_kV, filters = filter, 
                                            th_in_deg=th_deg, max_number_of_energy_bins=50,
                                            mAs=mAs, z=SDD_cm)   
        else:
            photons = 20000
            gvxr.setMonoChromatic(voltage_kV, "keV", photons)
            gvxr.setmAs(mAs)
            gvxr.setTubeAngle(th_deg)
            print("Number of incident photons = " + str(np.round(photons, 2)))

        if noise:  
            gvxr.enablePoissonNoise()
        else:
            gvxr.disablePoissonNoise()


    def setup_detector(self, xPixels ,yPixels, pixel_size = 0.1, scintillator=('CdWO4', 525), epsilon=0):
        """
        Creates the detector of size (xPixels x yPixels)/scaling. 

        :param xPixels: Number of pixels in the x-direction
        :param yPixels: Number of pixels in the y-direction
        :param pixel_size: The size of each pixel in the detector (in mm)
        :param scintillator: The scintillator (material, thickness (um)) for the detector
        :param epsilon: The std. deviation of gaussian line spread function (generating blur in detector)
        """
        gvxr.setDetectorUpVector(0,0,-1)
        gvxr.setDetectorNumberOfPixels(int(xPixels/self.scaling), int(yPixels/self.scaling))
        gvxr.setDetectorPixelSize(pixel_size * self.scaling, pixel_size * self.scaling, self.lu)

        if scintillator != None:
            scint_mat, scint_thick = scintillator
            gvxr.setScintillator(scint_mat, scint_thick, "um")
        
        if epsilon > 0:
            x = np.linspace(-2,2,41)
            coeff1 = 1.0/np.sqrt(2.0*(np.pi)*np.absolute(epsilon))
            coeff2 = -(np.power(x, 2)/(2.0*epsilon))
            lsf = coeff1*np.exp(coeff2)
            lsf /= np.sum(lsf)
            gvxr.setLSF(lsf)
        else: 
            gvxr.clearLSF()

        self.detector_height = yPixels/self.scaling        
        self.detector_width = xPixels/self.scaling

    def addMesh(self, label:str, path:str, density:float, center=False, element=None, compound=None, mixture = None, mass_fraction = None, unit = "mm"): 
        """
        Adds a mesh to the virtual XCT environment.

        :param label: A label that refers to the mesh.
        :param path: The path to the mesh-file.
        :param density: Density of the of the mesh (in g/cm3)
        :param center: Controls if the mesh is centered (True/False)
        :param element: The element of the mesh (only the mesh should be simulated as a single element, otherwise None), ex. "Al", "Fe"
        :param compound: The compound of the mesh (only the mesh should be simulated as a compound, otherwise None), ex. "H2O"
        :param mixture: The elemental mixture of the mesh, ex. ["H", "O", "N"]
        :param mass_fraction: (Only if mixture != None), The mass fraction of the applied mixture, ex. [0.4, 0.32, 0.28] 
        :param unit: The units in which the mesh is imported.
        """

        gvxr.loadMeshFile(label, str(path), unit)
        if center:
            gvxr.moveToCenter(label)
        
        if element != None:
            gvxr.setElement(label, element)
        if compound != None: 
            gvxr.setCompound(label, compound)
            gvxr.setDensity(label, density, "g/cm3")
        if mixture is not None: 
            gvxr.setMixture(label, mixture, mass_fraction)
            gvxr.setDensity(label, density, "g/cm3")

    def viewScene(self):
        gvxr.displayScene()
        gvxr.renderLoop()


    def compute2D(self):
        """
        Computes a 2D X-ray image. 

        :param save: to be implemented, saves the image if True

        :returns: The 2D X-ray image
        """
        if self.poly:
            img = np.array(gvxr.computeXRayImage(), dtype=np.single) / gvxr.getTotalEnergyWithDetectorResponse();
        else:
            img = gvxr.computeXRayImage()
        return img
    

    def computeCTAcquisition(self, projections, rot_center=[0,0,0], FFC=0, label=""):
        upVector = (0,0,-1)
        if label == "":
            self.CT_path = os.path.join(self.output_path, "CT-Scans/CT_" + str(projections)+ "_FFC=" + str(FFC))
        else:
            self.CT_path = os.path.join(self.output_path, "CT-Scans/CT_" + str(projections) + "_" + label + "_FFC=" + str(FFC))

        os.makedirs(self.CT_path, exist_ok=False)
        projections_path = os.path.join(self.CT_path, "Projections")

        start_acq = time.time()
        gvxr.computeCTAcquisition(projections_path, # the path where the X-ray projections will be saved.
                                                # If the path is empty, the data will be stored in the main memory, but not saved on the disk.
                                                # If the path is provided, the data will be saved on the disk, and the main memory released.
                                "",
                                projections, # The total number of projections to simulate.
                                0, # The rotation angle corresponding to the first projection.
                                False, # A boolean flag to include or exclude the last angle. It is used to calculate the angular step between successive projections.
                                360,
                                FFC, # The number of white images used to perform the flat-field correction. If zero, then no correction will be performed.
                                *rot_center, # The location of the rotation centre.
                                "mm", # The corresponding unit of length.
                                *upVector, # The rotation axis
                                True # If true the energy fluence is returned, otherwise the number of photons is returned
                                    # (default value: true)
        )
        end_acq = time.time()
        print("CT-Acquisition complete! Execution time: "+ str(np.round(end_acq-start_acq, 4)) + " s")

        json_fname = os.path.join(self.CT_path, "simulation-" + str(projections) + ".json")
        gvxr2json.saveJSON(json_fname)

        return json_fname


    def computeReconstruction(self, CT_path:str, json_fname:str, filter=filter):
        """
        Computes the 3D reconstruction using the FDK algorithm (for CBCT). 

        :param CT_path: The path to the folder in which the reconstruction volume will be saved.
        :param json_name: Path the the json-file containing the correct geomtry to perform the reconstruction, 
        this was created during the acquisition 
        :param filter: The filter used during the reconstruction. ex: "no filter", "ram-lak", "shepp-logan", "cosine", "hamming", "hann", or own numpy array

        :returns: The FDK reconstrucion in the form of an array
        """
        reader = JSON2gVXRDataReader(json_fname)
        data = reader.read()

        data = TransmissionAbsorptionConverter(white_level=data.max())(data) #Converts the XCT data by normlisation and then take the negative logarithm
        data.reorder(order='tigre') #Converts data_negative to TIGRE's required format
        data_image_geometry = data.geometry.get_ImageGeometry()

        #If no filtration wanted
        if filter == "no filter":
            n = FDK(data, data_image_geometry).get_filter_array().shape[0]
            filter = np.ones(n, dtype=np.float32)

        start_recon = time.time()
        self.reconstruction = FDK(data, data_image_geometry, filter=filter).run()
        end_recon = time.time()
        print("FDK-reconstruction complete! Execution time: " + str(np.round(end_recon-start_recon,4)) + " s")

        recon_np = self.reconstruction.as_array()

        stack_path = os.path.join(CT_path, f"Tiff-Stack_Filter={filter}.tif")

        tiff.imwrite(stack_path, recon_np)

        return recon_np
    

    def create_tiff_stack(self, stack_output_path, compression=None):
        """
        Creates a tiff-stack in the folder "stack_output_path" containing all vertical slices of the 3D reconstruction.

        :param stack_output_path: The folder in which the tiff-stack is to be saved in
        :param compression: Compression of each saved tiff-file. ex: None, "uint8", or "uint16"
        """
        os.makedirs(stack_output_path, exist_ok=True)

        # Full prefix path (no extension)
        full_prefix = os.path.join(stack_output_path, "stack")

        # Create tiff-writer
        writer = TIFFWriter(
            data=self.reconstruction,
            file_name=full_prefix,
            compression=compression 
        )
        writer.write()

