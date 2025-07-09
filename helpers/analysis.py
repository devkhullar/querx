import numpy as np
import pandas as pd
from XRBID.WriteScript import WriteReg
from XRBID.DataFrameMod import Find
from XRBID.CMDs import FitSED

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

def change_columns(df, columns=columns):
    df = df.rename(columns=columns)
    return df

def remove_unnamed(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def look_at_df(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # If needed, control the width of columns to avoid line wrapping
    pd.set_option('display.width', 1000)
    # If needed, adjust the max column width
    pd.set_option('display.max_colwidth', None)

    display(df)

def compare_dfs(df1, df2, 
                condition1, condition2):
    '''Compare two dataframes and create a new dataframe out
    of the first dataframe based on the given conditions.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe that will be used to create the new dataframe
    df2 : pd.DataFrame
        Dataframe to be compared with.
    condition1, condition2 : str or list
        Conditions to be used to compare the dfs. If list, pass 2
        conditions and the conditions must be the string headers that 
        will be compared between the dataframes.

    '''
    
    temp = pd.DataFrame()
    if isinstance(condition1, list):
        assert(len(condition1) == 2)
        condition1a = condition1[0]
        condition1b = condition1[1]
    else: 
        condition1a, condition1b = condition1, condition1
    if isinstance(condition2, list):
        assert(len(condition2) == 2)
        condition2a = condition2[0]
        condition2b = condition2[1]
    else: 
        condition2a, condition2b = condition2, condition2

    for index1, row1 in df1.iterrows():
        for index2, row2 in df2.iterrows():
            condition1 = (df1[condition1a][index1] == df2[condition1b][index2])
            condition2 = (df1[condition2a][index1] == df2[condition2b][index2])
            if condition1 and condition2:
                temp = temp._append(df1.iloc[index1], ignore_index=True)

    return temp
