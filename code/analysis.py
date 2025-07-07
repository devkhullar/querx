import numpy as np
import pandas as pd
from XRBID.WriteScript import WriteReg
from XRBID.DataFrameMod import Find
from XRBID.CMDs import FitSED

chandra_jwst_dir = "/Users/undergradstudent/Research/XRB-Analysis/Galaxies/M66/Chandra-JWST/"
input_model='/Users/undergradstudent/Research/XRB-Analysis/jwst-models/isochrone-query-step-0_009.dat'
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
        saved if `save_extinction` is set to True.
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
                # min_model = Find(bestfit, f'Test Statistic = {min(bestfit['Test Statistic'])}').head(1)
                absorption = pd.concat((absorption, min_model))
            except:
                pass
    if cols: absorption = absorption[cols].reset_index(drop=True)

    # get coordinates to create region files
    absorption = get_coords(absorption, df_photometry)
    if save_df: 
        absorption.to_csv(chandra_jwst_dir+'M66_XRB_dust_extinction.frame',
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


def get_coords(extinction_df, coords_df):
    '''Returns the dataframe with the associated RA and Dec coordinates
    of the XRB candidates.
    '''
    temp = extinction_df.copy()
    temp['RA'] = ''
    temp['Dec'] = ''
    for index, row in coords_df.iterrows():
        for ind, ro in temp.iterrows():
            condition1 = (coords_df['CSC ID'][index] == temp['CSC ID'][ind])
            condition2 = (coords_df['StarID'][index] == temp['StarID'][ind])
            if condition1 and condition2:
                temp['RA'][ind] = coords_df['RA'][index]
                temp['Dec'][ind] = coords_df['Dec'][index]
    
    return temp

def find_best_stars(df_photometry, df_notes):
    '''Compares two dataframes to only select the photometry of the best stars.
    '''

    temp = pd.DataFrame()
    for index, row in df_notes.iterrows():
        for index1, row1 in df_photometry.iterrows():
            if df_notes['CSC ID'][index] == df_photometry['CSC ID'][index1]:
                if df_notes['Best Star'][index] == df_photometry['StarID'][index1]:
                    temp = temp._append(df_photometry.iloc[index1], ignore_index=True)
                    
    temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]

    return temp

def change_columns(df, columns=columns):
    df = df.rename(columns=columns)
    return df

