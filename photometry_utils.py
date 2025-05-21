import matplotlib.pyplot as plt 
import numpy as np 
import time

from astropy.stats import sigma_clipped_stats
from astropy.io import fits

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

def find_objects(data, fwhm, threshold=None, sigma=5, conversion=0.031, radius=5, cmap='gray_r', vmin=0, vmax=0.3, std_multiple=5, aperture_color='#0547f9', plot=True):
    """
    Find objects in an image using the DAOFind algorithm and then plot apertures around the objects. 

    Note: this is under active construction
    Feature to add : 1. conversion rate on the basis of the filter used. 
                     2. I want the code to accept both data + hdu files
                     3. add conversion rate for different filters 

    
    PARAMETERS
    ----------
    data : data
        Data
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
    starttime = time.time()
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

    return objects

# Create Apertures
def create_apertures(positions, rad_list=(1, 31)):
    ap_rads = [i for i in range(1,31)]
    apertures_full = [CircularAperture(positions, r=r) for r in ap_rads]
    apertures_source = CircularAperture(positions, r=3) # 3px aperture photometry used for sources by default
    apertures_extended = CircularAperture(positions, r=10) # aperture photometry for clusters (default is 10 pixels)
    return apertures_full, apertures_source, apertures_extended

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


