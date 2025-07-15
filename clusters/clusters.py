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

hmxbs.crossref(
    cluster_region=chandra_hst_dir+"M66_assoc1_catalog_ws16pc_fk5.reg",
    cluster_name='Compact Association',
    search_radius=0.0005,
    coordsys='fk5',
    coordheads=['RA', 'Dec'],
    outfile='Test.txt'
)
print(hmxbs)

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

print(hmxbs)

# Merging the ages of the clusters to the dataframes
hmxbs.df = pd.merge(hmxbs.df, compact_association, left_on='Compact Association ID',
                    right_on='reg_id', how='left')
hmxbs.df = pd.merge(hmxbs.df, M66_best, )
hmxbs.df

hmxbs.calculate_velocity(
    velocity_err_headers=['2Sig', 'Age Err'],
    velocity_headers=['Distance', 'Age'],
    calc_err=True
)

hmxbs.make_regions(
    outfile='/Users/undergradstudent/Downloads/test_code.reg',
    additional_coords=['Compact Association RA', 'Compact Association Dec'],
    idheader='Compact Association ID'
)