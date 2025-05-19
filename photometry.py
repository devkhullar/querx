import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats

from photutils.detection import DAOStarFinder
from photutils.psf import fit_fwhm   
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, aperture_photometry

from perform_photometry import find_objects

from XRBID.AutoPhots import SubtractBKG, CorrectAp

import time

import os 
cd = os.chdir

long_filter = ["F070W", "F090W", "F115W", "F140M",
                    "F150W2", "F150W", "F162M", "F164N",
                    "F182M", "F187N", "F200W", "F210M",
                    "F212N", "F250M", "F277W", "F300M",
                    "F322W2", "F323N", "F335M", "F356W",
                    "F360M", "F405N", "F410M", "F430M",
                    "F444W", "F460M", "F466N", "F470N"
                    "F480M"]

short_filter = ["F070W", "F090W", "F115W", "F140M",
                "F150W2", "F150W", "F162M", "F164N",
                "F182M", "F187N", "F200W", "F210M",
                "F212N"]

'''
1. DAOFind to identify stars [done]
2. Generate background subtracted hdu image
3. Run Photometry using the background-subtracted image.  
        - full aperture photometry between 1-30 pixels radii
        - aperture photometry within 3 pixels 
        - aperture photometry within the extended radius for clusters 
4. Aperture correction on full aperture photometry. Apply correction
    to both the minimum aperture photometry and the extended aperture photometry 
'''

jwstdir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/JWST/"
f200w = jwstdir+"hlsp_phangs-jwst_jwst_nircam_ngc3627_f200w_v1p1_img.fits"

# find objects in the image
data = fits.getdata(f200w)
data = data[3200:3400, 3600:3800]
objects = find_objects(data, fwhm=0.17, vmax=10)

print("Object Identification worked")

# Background subtraction
data_sub = SubtractBKG(data)
print("Background Subtracted")
positions = np.transpose((objects['xcentroid'], objects['ycentroid']))
print("Background substraction worked")

# Create Apertures
def create_apertures(positions, rad_list=(1, 31)):
    ap_rads = [i for i in range(1,31)]
    apertures_full = [CircularAperture(positions, r=r) for r in ap_rads]
    apertures_source = CircularAperture(positions, r=3) # 3px aperture photometry used for sources by default
    apertures_extended = CircularAperture(positions, r=10) # aperture photometry for clusters (default is 10 pixels)
    return apertures_full, apertures_source, apertures_extended

apertures_full, apertures_sources, apertures_extended = create_apertures(positions)

print("Aperture creation worked...")

print("Starting Photometry...")
gal = "M66"
instrument = "nircam"
suffix = ''
filter = 'f200w'

def perform_photometry(data_sub, apertures, type, savefile=True):
    '''
    A helper function to calculate the aperture photometry.
    type : full/extended/sources
    '''
    starttime = time.time()
    photometry = aperture_photometry(data_sub, apertures, method='center')
    endtime = time.time()
    photometry.write("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_"+type+suffix+".ecsv", overwrite=True)
    print("photometry_"+gal+"_"+filter.lower()+"_"+instrument.lower()+"_"+type+suffix+".ecsv", "saved")
    return photometry

cd(jwstdir)
phot_full = perform_photometry(data_sub, apertures_full, type='full')
phot_sources = perform_photometry(data_sub, apertures_sources, type='sources')
phot_extended = perform_photometry(data_sub, apertures_extended, type='extended')
print("Aperture Photometry worked")

# Aperture Corrections
cd("/Users/undergradstudent/Research/XRB-Analysis/Notebooks")
short_EEFs = pd.read_csv('Encircled_Energy_SW_ETCv2.csv')
long_EEFs = pd.read_csv('Encircled_Energy_LW_ETCv2.csv')
# Use 20 pixels: 0.6'' for short wavelength and 1.6'' for long wavelength for  as the default
def my_EEF(filter):
    if filter in short_filter:
        return long_EEFs.at[14, filter]
    if filter in long_filter:
        return long_EEFs.at[19, filter]
    
filter = "F200W"
EEF = my_EEF(filter)
print("Aperture corrections...")


print("Aperture corrections...")
min_rad = 3
max_rad = 20
extended_rad = 10 
num_stars = 20
ap_rads = [i for i in range(1,31)]
zmag = np.mean([25.55, 25.56, 25.60, 25.66])
apcorrections = CorrectAp(phot_full, radii=ap_rads, EEF=EEF, num_stars=num_stars, zmag=zmag, \
                        min_rad=min_rad, max_rad=max_rad, extended_rad=extended_rad)
if len(apcorrections) > 0:
    apcorr = apcorrections[0]
    aperr = apcorrections[1]
    apcorr_ext = apcorrections[2]
    aperr_ext = apcorrections[3]
else: 
    apcorr = 0
    aperr = 0
    apcorr_ext = 0
    aperr_ext = 0

print(apcorr, aperr, apcorr_ext, aperr_ext)
