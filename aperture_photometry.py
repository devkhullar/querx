import matplotlib.pyplot as plt 
import numpy as np 

from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

def find_objects(data_file, fwhm, threshold=None, sigma=5, conversion=0.031, radius=5, cmap='viridis', vmin=0, vmax=0.3, aperture_color='limegreen'):
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
    sigma : float
        sigma to be used for finding the standard deviation.
    conversion : float
        The convesion from arcseconds to pixel.
        Default is 0.031.
    radius : float 
        Radius of the aperture to be applied on the sources
        detected. Default is 5.
    cmap : string
        Colour map to be used when plotting the image. Default 
        is viridis. 
    vmin : float
        vmin for the image. Default is 0
    vmax : float
        vmax for the image. Default is 0.3
    aperture_color : string
        The color of the apertures.

    RETURN
    ------
    positions : np.arrays
        The positions of the point sources.
        Also plots the apertures around the point 
        sources on the image. 
        """
    data = fits.getdata(data_file)
    mean, median, std = sigma_clipped_stats(data, sigma=5)

    # If a threshold is supplied manually
    if threshold:
        daofind = DAOStarFinder(fwhm=fwhm/conversion, threshold=threshold)
        objects = daofind(data)
        positions = np.transpose((objects["xcentroid"], objects["ycentroid"]))
    else:
        daofind = DAOStarFinder(fwhm=fwhm/conversion, threshold=5*std)
        objects = daofind(data)
        positions = np.transpose((objects["xcentroid"], objects["ycentroid"]))

    # Create apertures around sources
    apertures = CircularAperture(positions, r=radius)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    apertures.plot(color=aperture_color)
    plt.show()
    
    return positions