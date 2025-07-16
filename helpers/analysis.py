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

def find_absorption(
    df_photometry,
    df_notes,
    plotSED=False,
    cols=cols,
    create_regions=False,
    outfile=None,
    fontsize=20,
    instrument='nircam',
    min_models=10,
    input_model=input_model,
    idheader='StarID',
    radius=0.09,
    radunit='arcsec',
    color='blue',
    save_df=False,
    suffix=None
):
    '''
    Performs SED fitting on XRB sources and returns the dataframe with the model
    with the smallest test statistic.

    PARAMETERS
    ----------
    df_photometry.   : pd.DataFrame
        Dataframe with the photometric measurements.
    df_notes         : pd.DataFrame
        Dataframe with the header names (CSC ID in this case)
    save_df  : bool, False
        if True, saves the absorption dataframe.
    suffix           : str, None
        Suffix to add to the name of the dataframe that will be
        saved if `save_df` is set to True.
    RETURNS
    -------
    absorption    : pd.DataFrame
        DataFrame containing only the input columns information.  
    outfile       : Ds9 regions file
        Regions file containing the absorption of different XRB candidates

    '''
    photometry = df_photometry.copy()
    notes = df_notes.copy()
    absorption = pd.DataFrame()
    count = 0

    # to create a dataframe, absorption, with both the sourceid + the cscid
    for id in notes['CSC ID'].tolist():
        source = Find(photometry, f'CSC ID = {id}')
        for starid in source['StarID'].tolist():
            try:
                count += 1
                print(f"Count = {count}")
                print(f"Working on {id} {starid}")
                bestfit = FitSED(
                    df=Find(source, f'StarID = {starid}'),
                    instrument=instrument,
                    min_models=min_models,
                    plotSED=plotSED,
                    input_model=input_model,
                    model_ext=True,
                    idheader=idheader
                )
                bestfit['CSC ID'] = id
                print(bestfit)
                # Add the row with the smallest test statistic to absorption
                bestfit = bestfit.sort_values(by='Test Statistic')
                bestfit = bestfit.iloc[[0]]
                min_model = Find(bestfit, f'Test Statistic = {min(bestfit['Test Statistic'])}').iloc[[0]]
                absorption = pd.concat((absorption, min_model))
            except:
                pass
    if cols: absorption = absorption[cols].reset_index(drop=True)

    # get coordinates to create region files
    absorption = get_coords(absorption, df_photometry)
    if save_df: 
        absorption.to_csv(chandra_jwst_dir+f'M66_XRB_dust_extinction{suffix}.frame',
                          index=False)

    if create_regions:
        WriteReg(
            sources=absorption,
            outfile=outfile,
            coordsys='fk5',
            color=color,
            coordheads=['RA', 'Dec'],
            idheader='Av',
            radius=radius,
            radunit=radunit,
            fontsize=fontsize
        )

    return absorption

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
    ra, dec = df['RA'].values, df['Dec'].values
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
        # Incorporate unit conversion to also include arcsecs
        if unit_of_dist.lower() == 'arcsec':
            dist = dist * instrument_pixtoarcs[instrument]
        elif unit_of_dist.lower() == 'pc' or unit_of_dist.lower() == 'parsec':
            dist = dist * instrument_pixtoarcs[instrument] * arcsectopc
        elif unit_of_dist.lower() == 'km' or unit_of_dist.lower() == 'kilometer':
            # https://www.unitconverters.net/length/parsec-to-kilometer.htm
            dist = dist * instrument_pixtoarcs[instrument] * arcsectopc * u.pc.to(u.km)

        df[f'{catalog} Separation ({unit_of_dist})'] = dist
        object_id = f'{catalog} ID'
        object_ra = f'{catalog} RA'
        object_dec = f'{catalog} Dec'
        object_dist = f'{catalog} Separation ({unit_of_dist})'
        object_cols.extend([object_id, object_ra, object_dec, object_dist])

    if shorten_df: 
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

    # get the coordinates and IDs of the catalog being crossref'd
    df = get_coords(df=df, regions=regions, catalogs=catalogs)

    # Find the euclidean distance between the sources within the input dataframe
    # and the catalog being crossref'd
    df = euclidean_distance(
        df=df,
        catalogs=catalogs,
        imagefilename=imagefilename,
        instrument=instrument,
        arcsectopc=arcsectopc,
        **kwargs
    )

    return df

class XrayBinary:
    '''A class to study distances between XRBs and clusters along with doing
    further science with them'''
    def __init__(self, df):
        self.df = df.copy()
        if 'X' in self.df.columns:
            self.x = self.df['X'].values
        if 'Y' in self.df.columns:
            self.y = self.df['Y'].values
        if 'RA' in self.df.columns:
            self.ra = self.df['RA'].values
        if 'Dec' in self.df.columns:
            self.dec = self.df['Dec'].values

    def _repr_html_(self):
        return self.df._repr_html_()

    def __repr__(self):
        return self.df.__repr__() # Or self.df.to_string() for full display

    def __str__(self):
        return self.df.__str__() # Or self.df.to_string() for full display
    
    def crossref(
            self,
            cluster_region,
            cluster_name,
            sourceid='CSC ID',
            search_radius=0.005,
            coordsys='fk5',
            coordheads=['RA', 'Dec'],
            outfile='XRB_to_cluster.txt',
        ):
        '''A method to `Crossref` between XRBs and clusters.
        
        Parameters
        ----------
        cluster_region : path, str
            Path to the region file containing the cluster coordinates.
        cluster_name : list, str
            A list containing the name of the cluster.
        sourceid : str
            Header to use for the `Crossref`. Default is 'CSC ID'
        search_radius : float
            Search radius to `Crossref`. Will be in the same units as the `coordsys`.
            Default is 0.005
        coordheads: list, str
            A list of strings containing the coordinate headers. Default is 
            ['RA', 'Dec']
        outfile : path
            Path to download the output `Crossref`'d frame.
        
        Returns
        -------
        `Crossref`'d dataframe'''
        if isinstance(cluster_name, str): cluster_name = [cluster_name]
        if isinstance(cluster_region, str): cluster_region = [cluster_region]

        # Delete the headers related to the cluster_name from previous iterations of Crossref
        if f'{cluster_name[0]} ID' in self.df.columns:
            self.df.drop(f'{cluster_name[0]} ID', axis=1, inplace=True)
        if f'{cluster_name[0]} RA' in self.df.columns:
            self.df.drop(f'{cluster_name[0]} RA', axis=1, inplace=True)
        if f'{cluster_name[0]} Dec' in self.df.columns:
            self.df.drop(f'{cluster_name[0]} Dec', axis=1, inplace=True)

        self.df = Crossref(
            df=self.df,
            regions=cluster_region,
            catalogs=cluster_name,
            sourceid=sourceid,
            search_radius=search_radius,
            coordsys=coordsys,
            coordheads=coordheads,
            outfile=outfile,
        )

        return self.df

    def calculate_distance(
            self,
            filename,
            instrument,
            cluster_region,
            cluster_name,
            search_radius=0.0005,
            sourceid='CSC ID',
            coordsys='fk5',
            coordheads=['RA', 'Dec'],
            outfile='XRB_to_cluster.txt',
            frame='fk5',
            unit_of_coords='deg',
            unit_of_dist='km',
            arcsectopc=45.4,
            shorten_df=False,
            additional_cols=[]
    ):
        '''A method to calculate the distances between XRBs and clusters.
        
        This method is calculating the Euclidean norm by converting the input coordinates
        to pixels and then performing the euclidean norm on them. 

        Parametes
        ---------
        First few parameters the same as Crossref
        frame : str
            The coordinate reference frame to use to convert coordinates to pixels. 
            Default is 'fk5'.
        unit_of_coords : str,
            The units in which the coordinates are being input. Default is 'deg',
        arcsectopc : float
            The arcsec to parsec conversion. Default is 45.4 for the NGC 3626 galaxy
        shorten_df : bool
            If `True`, shortens the dataframe to only include the IDs and coordinates.
            Default is False
        additional_cols : list
            Additional columns to add to the output dataframe. Default is an empty list.

        Parameters
        ----------
        A dataframe containing the distances from star clusters to XRBs. If `calculate_velocity=True`,
        also finds the velocity of ejection of the XRBs from the candidate clusters.
        '''
        if isinstance(cluster_name, str): cluster_name = [cluster_name]
        if isinstance(cluster_region, str): cluster_region = [cluster_region]

        self.crossref(
            cluster_region=cluster_region,
            cluster_name=cluster_name,
            sourceid=sourceid,
            search_radius=search_radius,
            coordsys=coordsys,
            coordheads=coordheads,
            outfile=outfile
        )

        self._get_coords(cluster_region, cluster_name)

        self.df = euclidean_distance(
             filename=filename,
             df=self.df,
             catalogs=cluster_name,
             instrument=instrument,
             frame=frame,
             unit_of_coords=unit_of_coords,
             unit_of_dist=unit_of_dist,
             arcsectopc=arcsectopc,
             shorten_df=shorten_df,
             additional_cols=additional_cols
        )

        self.df = self.df.query(f'`{cluster_name[0]} ID`.notnull()').reset_index(drop=True)
        self.distance = self.df[f'Separation ({unit_of_dist})'].values
        return self.df
    
    def _get_coords(self, cluster_region, cluster_name):
        '''A helper method to extract coordinates.'''
        self.df = get_coords(self.df, cluster_region, cluster_name)
        return self.df
    
    def calculate_velocity(
            self,
            velocity_headers=['Distance', 'Age'],
            calc_err=False,
            velocity_err_headers=['Distance Err', 'Age Err']
    ):
        '''Calculate the maximum velocity of ejection of an XRB from a cluster.
        
        Note
        ----
        The input distance should be in km and the input age should be in Myrs.

        Parameters
        ----------
        velocity_headers : list(str)
            List of headers containing the distances and ages of the clusters
            respectively.
        calc_err : bool
            If `True`, finds the errors in the velocity of ejection of the
            XRB from the cluster.
        velocity_err_headers : list(str)
            A list of strings containing the headers of the errors in distance 
            and the age of the cluster respectively. 

        Returns
        ----------
        A dataframe containing the velocity of ejection of the X-ray Binary from
        the cluster. If `calc_err=True`, calculates the errors in the velocity.
        '''
        time = self.df[velocity_headers[1]].values * 31556952000000 # https://www.unitsconverters.com/en/Millionyears-To-Second/Unittounit-5988-91
        self.df['Velocity (km/s)'] = self.df[velocity_headers[0]] / time
        if calc_err:
            d, d_err = self.df[velocity_headers[0]], self.df[velocity_headers[0]]
            t, t_err = self.df[velocity_err_headers[1]], self.df[velocity_err_headers[1]]
            err = np.sqrt((d_err / d) ** 2 + (t_err / t) ** 2)
            self.df['Velocity Err (km/s)'] = self.df['Velocity (km/s)'] * err

        self.velocity = self.df['Velocity'].values
        return self.df
    
    def make_regions(
        self,
        outfile,
        coordsys='fk5',
        coordheads=['RA', 'Dec'],
        reg_type='ruler',
        additional_coords=False,
        idheader='CSC ID',
        color='blue',
        radius=1,
        radunit='arcsec',
        width=1,
        fontsize=10,

    ):
        WriteReg(
            sources=self.df,
            outfile=outfile,
            coordsys=coordsys,
            coordheads=coordheads,
            reg_type=reg_type,
            additional_coords=additional_coords,
            idheader=idheader,
            color=color,
            radius=radius,
            radunit=radunit,
            width=width,
            fontsize=fontsize
        )

    def plot_kde(self, x, bw_adjust,
                      label='Distance to the nearest cluster',
                      fill=True, alpha=0.5, **kwargs
                      ):
        sns.kdeplot(x=x, bw_adjust=bw_adjust,
                    fill=True, alpha=0.5,
                    **kwargs)
        plt.xlabel(label)
        plt.show()

    def plot_hist(self, x, bins=None, 
                  xlabel='Distance to the nearest cluster (km)',
                  ylabel='Density',
                   **kwargs):
        plt.hist(x=x, bins=bins, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
