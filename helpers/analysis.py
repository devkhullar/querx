import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from XRBID.DataFrameMod import Find
from XRBID.CMDs import FitSED
from XRBID.Sources import Crossref, GetCoords, GetIDs
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from helpers.regions import WriteReg
import astropy.units as u

chandra_jwst_dir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/Chandra-JWST/"
input_model = '/Users/undergradstudent/Research/XRB-Analysis/jwst-models/isochrone-query-step-0_009.dat'

# Default column names for creating the absorption dataframe
cols = [
        'logAge',
        'Mass',
        'Av',
        'logL',
        'logTe',
        'logg',
        'Test Statistic',
        'StarID',
        'CSC ID'
    ]

# default 
columns={
    'F200W': 'F200Wmag',
    'F300M': 'F300Mmag',
    'F335M': 'F335Mmag',
    'F360M': 'F360Mmag',
    'F200W Err': 'F200Wmag Err',
    'F300M Err': 'F300Mmag Err',
    'F335M Err': 'F335Mmag Err',
    'F360M Err': 'F360Mmag Err'
}

instrument_pixtoarcs = {
    'acs'     : 0.05,
    'wfc3'    : 0.03962,
    'nircaml' : 0.063,
    'nircams' : 0.031,
}

arcsectopc = 45.4 # https://iopscience.iop.org/article/10.3847/1538-4357/ace162

def remove_unnamed(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def get_coords(df, regions, catalogs):
    '''A helper function to extract the coordinates and IDs of clusters
    and merge it with the `Crossref`'d dataframe.'''
    for region, catalog in zip(regions, catalogs):
        temp = pd.DataFrame()
        ids = GetIDs(region, verbose=False)
        ra, dec = GetCoords(region, verbose=False)
        temp[f'{catalog} RA'] = ra
        temp[f'{catalog} Dec'] = dec
        temp[f'{catalog} ID'] = ids
        temp[f'{catalog} ID'] = temp[f'{catalog} ID'].astype(float)

        df = pd.merge(
            df,
            temp,
            how='left',
            on=f'{catalog} ID'
        )

    return df

def euclidean_distance(imagefilename,
                       df,
                       catalogs, 
                       instrument,
                       coordheads=['RA', 'Dec'],
                       frame='fk5', 
                       unit_of_coords='deg', 
                       unit_of_dist='km',
                       arcsectopc=45.4,
                       shorten_df=False,
                       additional_cols=[]
):
    '''Calculate euclidean distance between two sets of objects
    
    Parameters
    ----------
    filename : str
        The path of the base image to be used for distance calculation.
    df : pd.DataFrame
        Dataframe containing the coordinates of the objects to compare
    catalogs : list
        list containing the names of the objects being compared to.
        Default is 'fk5'
    instrument : str
    The instrument of the base image. Required to convert coordinates to coordinates
    other than pixels. Default is `None`. Other options include 'wfc3', 'acs', 
    'nircamL' for long wavelength with NIRCam and 'nircamS' for short wavelength
    with NIRCam.
    frame : str
        The reference coordinate frame of the object. Will be used to 
        convert coordinates to pixels. Default is 'fk5.
    unit_of_coords : str
        The units of the coordinates that are being extracted from the dataframe.
        Default is 'deg'
    unit_of_dist : str
        The units to use in the distances between the objects. Default is 'km'.
    arcsectopc : float
        The arcsec to parsec conversion to use for changing coordinates to coordinates. Default is 45.4''
        for the NGC 3627 galaxy.
        This parameter is not required to pass usually as the
        `instrument` parameter uses the `pixtoarcs` conversion based upon the instrument being used.
    shorten_df : bool
        If `True`, provides a smaller dataframe containing only the CSC ID, coordinates (image and others)
        as well as the distances. Default is False
    additional_cols : list of strings
        Additional columns to include in the shortened dataframe
    Returns
    -------
    df : pd.Dataframe
        Dataframe containing the distances between the objects
    '''
    df = df.copy()
    hdu = fits.open(imagefilename)
    try: wcs = WCS(hdu['SCI'].header)
    except: wcs = WCS(hdu['PRIMARY'].header)

    # If the dataframe contains the x and y coordinates
    # The code below has been commented because I suspect there is something going
    # wrong with the conversion between the coordinates. This is likely due to how 
    # data is stored within pandas dataframes and numpy arrays. I am still working on
    # how to combat that. Until then, convert the dataframes RA and Dec
    # if 'X' and 'Y' in df.columns:
    #     x, y = df['X'].values, df['Y'].values
    # else: 
    ra, dec = df[coordheads[0]].values, df[coordheads[1]].values
    x, y = SkyCoord(ra, dec, frame=frame, unit=unit_of_coords).to_pixel(wcs)

    arr = np.array([x, y]).T

    object_cols = []

    # Extract the coordinates that are going to be used to calculate 
    # the distance to the first object
    object_cols = []
    for catalog in catalogs:
        if (f'{catalog} X' and f'{catalog} Y') in df.columns:
            x1, y1 = df[f'{catalog} X'].values, df[f'{catalog} Y'].values
        else:
            ra1, dec1 = df[f'{catalog} RA'].values, df[f'{catalog} Dec'].values
            x1, y1 = SkyCoord(ra1, dec1, frame=frame, unit=unit_of_coords).to_pixel(wcs)

        # the comparison array contains the coordinates which will calculate
        # the distance of the first object to these objects.
        object_arr = np.array([x1, y1]).T

        dist = np.array([np.linalg.norm(arr[i] - object_arr[i]) for i in range(len(df))])
        # Save arcsec coords 
        dist = dist * instrument_pixtoarcs[instrument]
        df[f'{catalog} Separation (arcsecs)'] = dist
        # convert from arcsec to parsecs
        dist = dist * arcsectopc 
        df[f'{catalog} Separation (pc)'] = dist
        # convert from parsecs to km
        dist = dist * u.pc.to(u.km)

        df[f'{catalog} Separation ({unit_of_dist})'] = dist

    if shorten_df: 
        object_id = f'{catalog} ID'
        object_ra = f'{catalog} RA'
        object_dec = f'{catalog} Dec'
        object_dist = f'{catalog} Separation ({unit_of_dist})'
        object_cols.extend([object_id, object_ra, object_dec, object_dist])
        cols = ['CSC ID', 'X', 'Y', 'RA', 'Dec'] + object_cols + additional_cols
        df = df[cols].reset_index(drop=True)

    return df

def calculate_distance(
        df,
        regions,
        catalogs,
        search_radius,
        imagefilename,
        instrument='wfc3',
        coordsys='fk5',
        coordheads=['RA', 'Dec'],
        arcsectopc=45.5,
        sourceid='CSC ID',
        **kwargs
):
    df = df.copy()
    if isinstance(regions, str): regions = [regions]
    if isinstance(catalogs, str): catalogs = [catalogs]

    # Crossref to find the sources withing the given prom
    df = Crossref(
        df=df,
        regions=regions,
        catalogs=catalogs,
        search_radius=search_radius,
        coordsys=coordsys,
        coordheads=coordheads,
        sourceid=sourceid
    )
    print("Crossrefing done...")

    # get the coordinates and IDs of the catalog being crossref'd
    df = get_coords(df=df, regions=regions, catalogs=catalogs)

    # Find the euclidean distance between the sources within the input dataframe
    # and the catalog being crossref'd
    df = euclidean_distance(
        df=df,
        catalogs=catalogs,
        coordheads=coordheads,
        imagefilename=imagefilename,
        instrument=instrument,
        arcsectopc=arcsectopc,
        **kwargs
    )

    print("Done! Calculated Distances...")
    df = df.query(f'`{catalogs[0]} ID`.notnull()').reset_index(drop=True)
    return df

def calculate_velocity(df, coordheads, catalog='Cluster', errorheads=False,
                       shorten_df=False, idheader='ID', additional_cols=[],
                       distance_unit='km', time_unit='s'):
    '''Calculate the velocity of ejection of an XRB from a candidate host cluster.'''
    print("Make sure that the distances and ages are in km and seconds respectively...")
    df = df.copy()
    # The `astype(float)` method helped prevent a bug in the calculations
    dist, time = df[coordheads[0]].astype(float).values, df[coordheads[1]].astype(float).values
    df[f'Min {catalog} Velocity ({distance_unit}/{time_unit})'] = dist / time
    if errorheads:
        dist_err, time_err = df[errorheads[0]].astype(float).values, df[errorheads[1]].astype(float).values
        err = np.sqrt((dist_err / dist) ** 2 + (time_err / time) ** 2)
        df[f'Min {catalog} Velocity Err ({distance_unit}/{time_unit})'] = df[f'Min {catalog} Velocity (km/s)'].values * err
    
    if shorten_df and errorheads:
        cols = [idheader] + additional_cols \
                + [f'MIn {catalog} Velocity ({distance_unit}/{time_unit})',
                f'Min {catalog}  Velocity Err ({distance_unit}/{time_unit})']
        df = df[cols]
    elif shorten_df: 
        cols = [idheader] + additional_cols \
                + [f'Min {catalog} Velocity ({distance_unit}/{time_unit})']
        df = df[cols]

    return df

