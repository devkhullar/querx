# Date created: 8 July 2025
# author: @devkhullar
# Last updated: 
# Update description: 

'''An algorithm to find the distances between clusters and X-ray binaries'''

import numpy as np
import pandas as pd

from XRBID.DataFrameMod import Find
from XRBID.Sources import Crossref, GetCoords

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import os, sys
sys.path.insert(0, '/Users/undergradstudent/Research/XRB-Analysis/Notebooks')
cd = os.chdir

from helpers.analysis import remove_unnamed, XrayBinary
from helpers.regions import WriteReg

hstdir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/HST/"
chandra_hst_dir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/Chandra-HST/"
chandra_jwst_dir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/Chandra-JWST/"
jwstdir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/JWST/"
M66_dir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/"
f555w = hstdir+"M66_mosaic_uvis_f555w_drc_sci.fits"
# cluster catalogs from the PHANGS catalogs
cluster_cataglog_wfc3 = pd.read_csv(chandra_hst_dir+"M66_cluster_catalog_uvis_fk5.frame")
compact_assoc_16pc = pd.read_csv(chandra_hst_dir+"assoc_catalog_ws16pc.frame")           # uses 16 pc watershed algorithm 
compact_assoc_acs = pd.read_csv(chandra_hst_dir+"M66_assoc_catalog_acs-uvis.frame")

# uses 16 pc watershed algorithm 
compact_association = remove_unnamed(pd.read_csv(chandra_hst_dir+"assoc_catalog_ws16pc.frame")) 
compact_association = compact_association.rename(
    columns = {
        'reg_dolflux_Age_MinChiSq' : 'Age',
        'reg_dolflux_Age_MinChiSq_err' : 'Age Err'
    }
)

# uses 16 pc watershed algorithm 
compact_association = remove_unnamed(pd.read_csv(chandra_hst_dir+"assoc_catalog_ws16pc.frame")) 
compact_association = compact_association.rename(
    columns = {
        'reg_dolflux_Age_MinChiSq' : 'Age',
        'reg_dolflux_Age_MinChiSq_err' : 'Age Err'
    }
)
compact_association = compact_association[['reg_id', 'Age', 'Age Err']]
# dataframe containing all the point sources within the 2 sig of the chandra data  
daoclean = remove_unnamed(pd.read_csv(chandra_hst_dir+"M66_daoclean_matches.frame"))
# dataframe containing the classification and the best selected star within the 2sig
M66_notes = remove_unnamed(pd.read_csv(chandra_hst_dir+"M66_XRB_notes.txt"))  
M66_best = remove_unnamed(pd.read_csv(chandra_hst_dir+"M66_csc_bestrads.frame"))     
M66_best = M66_best[['CSC ID', '2Sig']]

# Dataframe containing only the HMXBs of M66
M66_hmxbs = M66_notes.query('Class == "HMXB"')
# Merge the classifications 
best_stars = pd.merge(daoclean, M66_hmxbs, left_on=['CSC ID', 'StarID'],
                      right_on=['CSC ID', 'Best Star'], how='right')

hmxbs = XrayBinary(best_stars)
hmxbs.calculate_distance(
    cluster_region=chandra_hst_dir+"M66_assoc1_catalog_ws16pc_fk5.reg",
    cluster_name='Compact Association',
    search_radius=0.0005,
    coordsys='fk5',
    filename=f555w,
    instrument='wfc3',
    shorten_df=True,
    additional_cols=['Best Star'],
    unit_of_dist='km',
)
hmxbs

print(hmxbs)

# Merging the ages of the clusters to the dataframes
hmxbs.df = pd.merge(hmxbs.df, compact_association, left_on='Compact Association ID',
                    right_on='reg_id', how='left')
hmxbs.df = pd.merge(hmxbs.df, M66_best, )
hmxbs.df

hmxbs.calculate_velocity(
    velocity_err_headers=['2Sig', 'Age Err'],
    velocity_headers=['Distance (km)', 'Age'],
    calc_err=True
)

hmxbs.make_regions(
    outfile='/Users/undergradstudent/Downloads/test_code.reg',
    additional_coords=['Compact Association RA', 'Compact Association Dec']
)

# print(f"Printing best stars \n{best_stars}")

# # regions to crossreference
# regions = [
#     chandra_hst_dir+"M66_cluster_catalog_fk5.reg",
#     chandra_hst_dir+"M66_assoc1_catalog_ws16pc_fk5.reg",
#     chandra_hst_dir+"M66_assoc_catalog_acs-uvis_fk5.reg"
# ]

# catalogs = ['cluster wfc3', 
#             'CA wfc3', 'CA acs'
#             ]

# # search_radius = 1000 / 45.4 # https://iopscience.iop.org/article/10.3847/1538-4357/ace162#apjace162s2
# search_radius = 0.0005
# print(f'Using search radius: {search_radius}')

# crossref = calculate_distances(
#     df=best_stars,
#     catalogs=catalogs,
#     regions=regions,
#     search_radius=search_radius,
# )
# print("This is the crossref'd df")
# print(crossref)

# # Sources that were matched with HST sources
# condition = '`cluster wfc3 RA`.notnull() or `CA wfc3 RA`.notnull() or `CA acs RA`.notnull()'
# matches = crossref.query(condition).reset_index(drop=True)
# print("Printing matches")
# print(matches)
# # print(matches.query('`cluster wfc3 ID`.notnull()'))

# print(matches['RA'][0])

# WriteReg(
#     sources=matches,
#     outfile='/Users/undergradstudent/Downloads/test_calc_distance.reg',
#     coordsys='fk5',
#     coordheads=['RA', 'Dec'],
#     reg_type='ruler',
#     additional_coords=['cluster wfc3 RA', 'cluster wfc3 Dec'],
#     idheader='CSC ID',
#     radunit='arcsec',
# )

# hdu = fits.open(f555w)

# def euclidean_distance(filename,
#                        df,
#                        catalogs, 
#                        frame='fk5', 
#                        unit_of_coords='deg', 
#                        unit_of_dist='arcsec',
#                        instrument=None,
#                        pixtoarcs=None,
#                        shorten_df=False,
#                        additional_cols=[]
# ):
#     '''Calculate euclidean distance between two sets of objects
    
#     Parameters
#     ----------
#     filename : str
#         The path of the base image to be used for distance calculation.
#     df : pd.DataFrame
#         Dataframe containing the coordinates of the objects to compare
#     catalogs : list
#         list containing the names of the objects being compared to.
#         Default is 'fk5'
#     frame : str
#         The reference coordinate frame of the object. Will be used to 
#         convert coordinates to pixels. Default is 'fk5.
#     unit_of_coords : str
#         The units of the coordinates that are being extracted from the dataframe.
#         Default is 'deg'
#     unit_of_dist : str
#         The units to use in the distances between the objects. Default is 'arcsec'.
#         Other option is 'pix'.
#     instrument : str
#         The instrument of the base image. Required to convert coordinates to coordinates
#         other than pixels. Default is `None`. Other options include 'wfc3', 'acs', 
#         'nircamL' for long wavelength with NIRCam and 'nircamS' for short wavelength
#         with NIRCam.
#     pixtoarcs : float
#         The pixel to arcsecond conversion to use for changing coordinates to coordinates other
#         than pixels. Default is `None`. This parameter is not required to pass usually as the
#         `instrument` parameter uses the `pixtoarcs` conversion based upon the instrument being used.
#     shorten_df : bool
#         If `True`, provides a smaller dataframe containing only the CSC ID, coordinates (image and others)
#         as well as the distances. Default is False
#     additional_cols : list of strings
#         Additional columns to include in the shortened dataframe

#     Returns
#     -------
#     df : pd.Dataframe
#         Dataframe containing the distances between the objects
#     '''
#     df = df.copy()
#     hdu = fits.open(filename)
#     try: wcs = WCS(hdu['SCI'].header)
#     except: wcs = WCS(hdu['PRIMARY'].header)

#     # If the dataframe contains the x and y coordinates
#     # The code below has been commented because I suspect there is something going
#     # wrong with the conversion between the coordinates. This is likely due to how 
#     # data is stored within pandas dataframes and numpy arrays. I am still working on
#     # how to combat that. Until then, convert the dataframes RA and Dec
#     # if 'X' and 'Y' in df.columns:
#     #     x, y = df['X'].values, df['Y'].values
#     # else: 
#     ra, dec = df['RA'].values, df['Dec'].values
#     x, y = SkyCoord(ra, dec, frame=frame, unit=unit_of_coords).to_pixel(wcs)

#     arr = np.array([x, y]).T

#     cmpr_cols = []

#     # Extract the coordinates that are going to be used to calculate 
#     # the distance to the first object
#     cmpr_cols = []
#     for catalog in catalogs:
#         if (f'{catalog} X' and f'{catalog} Y') in df.columns:
#             x1, y1 = df[f'{catalog} X'].values, df[f'{catalog} Y'].values
#         else:
#             ra1, dec1 = df[f'{catalog} RA'].values, df[f'{catalog} Dec'].values
#             x1, y1 = SkyCoord(ra1, dec1, frame=frame, unit=unit_of_coords).to_pixel(wcs)

#         # the comparison array contains the coordinates which will calculate
#         # the distance of the first object to these objects.
#         cmpr_arr = np.array([x1, y1]).T

#         dist = np.array([np.linalg.norm(arr[i] - cmpr_arr[i]) for i in range(len(df))])
#         # Incorporate unit conversion to also include arcsecs
#         if unit_of_dist == 'arcsec':
#             if instrument:
#                 if instrument.lower() == 'acs':
#                     pixtoarcs = 0.05
#                 elif instrument.lower() == 'wfc3':
#                     pixtoarcs = 0.03962
#                     dist = dist * pixtoarcs
#                 elif instrument.lower() == 'nircaml': # if Nircam long wavelength
#                     pixtoarcs = 0.063
#                     dist = dist * pixtoarcs
#                 elif instrument.lower() == 'nircams': # if Nircam short wavelength
#                     pixtoarcs = 0.031
#                     dist = dist * pixtoarcs
#             elif pixtoarcs:
#                     if not pixtoarcs: input("Please input pixtoarcs") 
#                     dist = dist * pixtoarcs

#         df[f'Distance to {catalog} ({unit_of_dist})'] = dist

#         cmpr_cols.extend([f'{catalog} RA', f'{catalog} Dec', f'Distance to {catalog} ({unit_of_dist})'])
#         print(cmpr_cols)
#     if shorten_df: 
#         cols = ['CSC ID', 'X', 'Y', 'RA', 'Dec'] + cmpr_cols + additional_cols
#         df = df[cols].reset_index(drop=True)

#     return df

# dist_df = euclidean_distance(hdu, matches, catalogs, unit_of_dist='arcsec', 
#                              pixtoarcs=0.03962,
#                              shorten_df=True)

# print(dist_df)



