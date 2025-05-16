import matplotlib.pyplot as plt 
import numpy as np 
import time

from astropy.stats import sigma_clipped_stats
from astropy.io import fits

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

def find_objects(hdu, fwhm, threshold=None, sigma=5, conversion=0.031, radius=5, cmap='gray_r', vmin=0, vmax=0.3, std_multiple=5, aperture_color='#0547f9', plot=True):
    """
    Find objects in an image using the DAOFind algorithm and then plot apertures around the objects. 
    Feature to add : conversion rate on the basis of the filter used. 
    
    PARAMETERS
    ----------
    data_file : path
        Path to where data is stored.
    fwhm : float
        Full Width at Half Maximum of a star in arcseconds. 
        This code will automatically convert the fwhm from 
        arcseconds to pixels.
    threshold : float
        If specified, it will override the threshold to be
        used for identifying stars. Otherwise, the threshold
        is used as a multiple of the standard deviation found
        through the sigma clipped statistics on the data.
    sigma : int, default=5
        sigma to be used for finding the standard deviation.
    conversion : float, default=0.031
        The convesion from arcseconds to pixel.
        Default is 0.031 for JWST NIRCam short wavelength
        filters.
    radius : float, default=5 
        Radius of the aperture to be applied on the sources
        detected. Default is 5.
    cmap : string, default='viridis'
        Colour map to be used when plotting the image. Default 
        is viridis. 
    vmin : float, default=0
        vmin for the image. Default is 0
    vmax : float, default=0.3
        vmax for the image. Default is 0.3
    aperture_color : string, default='limegreen'
        The color of the apertures.

    RETURN
    ------
    positions : np.arrays
        The positions of the point sources.
        Also plots the apertures around the point 
        sources on the image. 
        """
    # try: data = hdu['SCI',1].data
    # except: data = hdu['PRIMARY',1].data
    starttime = time.time()
    data = fits.getdata(hdu)
    mean, median, std = sigma_clipped_stats(data, sigma=sigma)

    # If threshold is provided manually.
    if threshold:
        daofind = DAOStarFinder(fwhm=fwhm/conversion, threshold=threshold)
        objects = daofind(data)
    else:
        daofind = DAOStarFinder(fwhm=fwhm/conversion, threshold=std_multiple*std)
        objects = daofind(data)

    print("Found", len(objects), "objects.")
    positions = np.transpose((objects["xcentroid"], objects["ycentroid"]))

    # Create apertures around sources
    if plot:
        apertures = CircularAperture(positions, r=radius)
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        apertures.plot(color=aperture_color)
        plt.show()
    
    endtime = time.time()
    print("Time for the run:", (endtime-starttime)/60., "minutes")

    return positions


def find_objects_in_subset(data, fwhm, threshold=None, sigma=5, conversion=0.031, radius=5, cmap='gray_r', vmin=0, vmax=0.3, std_multiple=5, aperture_color='limegreen', plot=True):
    mean, median, std = sigma_clipped_stats(data, sigma=sigma)
    if threshold:
        daofind = DAOStarFinder(fwhm=fwhm/conversion, threshold=threshold)
        objects = daofind(data)
        
    else:
        daofind = DAOStarFinder(fwhm=fwhm/conversion, threshold=std_multiple*std)
        objects = daofind(data)

    print("Found", len(objects), "objects.")
    positions = np.transpose((objects["xcentroid"], objects["ycentroid"]))

    # Create apertures around sources
    if plot:
        apertures = CircularAperture(positions, r=radius)
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        apertures.plot(color=aperture_color)
        plt.show()
    
    return positions

# Test 1
jwstdir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/JWST/"
f200w = jwstdir+"hlsp_phangs-jwst_jwst_nircam_ngc3627_f200w_v1p1_img.fits"
# data = fits.getdata(f200w)
fwhm = 0.17
objects = find_objects(f200w, fwhm=fwhm, sigma=5, vmax=10, std_multiple=3)

# jwstdir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/JWST/"
# f200w = jwstdir+"hlsp_phangs-jwst_jwst_nircam_ngc3627_f200w_v1p1_img.fits"
# data = fits.getdata(f200w)
# data = data[2000:4000, 2000:4000]
# fwhm = 0.10*2
# objects = find_objects_in_subset(data, fwhm=fwhm, sigma=5, vmax=10, std_multiple=1)