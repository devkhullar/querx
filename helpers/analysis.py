import numpy as np
import pandas as pd
from XRBID.WriteScript import WriteReg
from XRBID.DataFrameMod import Find
from XRBID.CMDs import FitSED
from XRBID.Sources import Crossref, GetCoords

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

    Example
    -------
    best_stars = compare_dfs(daoclean, M66_sources, 
                         'CSC ID', ['StarID', 'Best Star'])
    >>> will loop through the dfs to compare CSC IDs and then compare
    >>> the first dataframe's 'StarID' property with the second frame's
    >>> 'Best Star' header
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

def Crossref(df=None, regions=False, catalogs=False, coords=False, sourceid="ID", search_radius=3, coordsys="img", coordheads=False, verbose=True, shorten_df=False, outfile="crossref_results.txt"): 

	"""

	UPDATE NEEDED: Keep the other columns in the dataframe.

	From input DataFrame and/or region files (in image coordinate format), finds overlaps within a given 
	search radius of the DataFrame sources and prints all ID names to a file as a DataFrame. 
	If the coordinates are given as ['RA', 'Dec'] instead of ['X','Y'], must change coordsys from "img" to "fk5" 
	and convert search_radius from pixels to degrees. Can feed in the name of the catalogs used to output 
	as DataFrame headers. Otherwise, the region name will be used.

	NOTE: There is an error in this where if the first region file doesn't have a counterpart in the first 
	entry of the overlap file, the first entry may be split into multiple entries. Check file.

	PARAMETERS
	-----------
	df		[pd.DataFrame]	: DataFrame containing the coordinates of the sources for which the counterparts 
					  will be found in the given region files or catalogs. 
	regions 	[list]		: List of filenames for regions to cross-reference sources from. This should be
					  in the same coordinate system as the units in df. 
	catalogs	[list]		: Name of the catalogs associated with the input region files. This will be used to
					  define the ID header for sources in each region file. If none is given, then the  
					  region file name is used as the respective source ID header.
	coords 		[list]		: List of coordinates to cross-reference; can be given instead of regions. 
	sourceid	[str] ('ID')	: Name of header containing the ID of each source in df. 
	search_radius	[list] (3)	: Search radius (in appropriate units for the coordinate system) around each source in df. 
					  Can be read in as a single value or a list of values (for unique radii).
	coordsys	[str] ('img')	: Coordinate system of the region files. NOTE: there may be issues reading in 'fk5'. 
				   	  'img' (pixel) coordinates are recommended. 
	coordheads	[list]		: Name of header under which coordinates are stored. Will assume ['X','Y'] or ['x','y'] if coordsys='img'
					  or ['RA','Dec'] if coordsys is 'fk5'. 
	verbose 	[bool] (True)	: Set to False to avoid string outputs. 
	shorten_df	[bool] (False)	: If True, shortens the output DataFrame to only include the original ID of each source in df, 
					  the coordinates, and the counterpart IDs of each of the other catalogs. Otherwise, will maintain
					  the original headers of the input DataFrame df.
	outfile		[str]		: Name of output file to save matches to. By default, saves to a file called 'crossref_results.txt'

	RETURNS
	---------
	Matches		[pd.DataFrame]	: DataFrame containing the original ID of each source, its coordinates, and the ID of all 
					  corresponding matches in each of the input region files or coordinates. 
	
	"""

	sources = df.copy()

	xlist = []
	ylist = []
	idlist = []

	# headerlist keeps track of the headers in the input DataFrame
	# if shorten_df = False, will use this to reapply addition headers to the Matches DataFrame
	dfheaderlist = sources.columns.tolist()
	
	# Removing headers that will be duplicated later
	dfheaderlist.remove(sourceid)

	if not isinstance(search_radius, list): search_radius = [search_radius]*len(sources)

	masterlist = [] # list of all matched sources

	if regions:
		if not isinstance(regions, list): regions = [regions]
		for i in regions: 
			idlist.append(GetIDs(i, verbose=False))
			xtemp, ytemp = GetCoords(i, verbose=False)
			xlist.append(xtemp)
			ylist.append(ytemp)
	elif coords: 
		# if given coords, they should be read in as a list of [xcoords, ycoords]. 
		xlist = coords[0]
		ylist = coords[1]
		if not isinstance(xlist, list): 
			xlist = [xlist]
			ylist = [ylist]

	blockend = 0 # keeps track of the index of the current 'block' of counterparts associated with a single base source 

	# Figuring out the coordinate headers if not given
	if not isinstance(coordheads, list): 
		coordheads = [False, False]
		if coordsys == "fk5": 
			if "RA" in dfheaderlist: coordheads[0] = "RA" 
			elif "ra" in dfheaderlist: coordheads[0] = "ra" 
			if "Dec" in dfheaderlist: coordheads[1] = "Dec" 
			elif "dec" in dfheaderlist: coordheads[1] = "dec"
		elif coordsys == "img":
			if "X" in dfheaderlist: coordheads[0] = "X"
			elif "x" in dfheaderlist: coordheads[0] = "x"
			if "Y" in dfheaderlist: coordheads[1] = "Y" 
			elif "y" in dfheaderlist: coordheads[1] = "y" 
		if not coordheads[0] and not coordheads[1]: # if coordinates not found, prompt user to input 
			coordheads = input("Coordinate headers not found. Please input headers separated by comma (xhead,yhead): ")
			coordheads = [i.strip() for i in coordheads.split(",")]
	
	# Removes headers from list, to avoid duplications later
	dfheaderlist.remove(coordheads[0])
	dfheaderlist.remove(coordheads[1])

	if verbose: print("Finding cross-references between sources. This will take a few minutes. Please wait.. ")
	for i in range(len(sources)): # for each source in the DataFrame
		# Properties of each source
		# Pulling the coordinates of each source
		xtemp = sources[coordheads[0]][i]
		ytemp = sources[coordheads[1]][i]

		tempid = sources[sourceid][i]
		tempn = 0  
		# tempn keeps track of the number of overlap sources identified in the current list for the current base source (used as index)

		# Search area around each source
		tempxmax = xtemp+search_radius[i]
		tempxmin = xtemp-search_radius[i]
		tempymax = ytemp+search_radius[i]
		tempymin = ytemp-search_radius[i]

		# Adding each new source to the list, starting as a list of "None" values
		# If no counterparts are found, the source will appear with "None" for counterparts.
		tempids = [None]*(len(idlist) + 1)
		tempids[0] = tempid 			# adding original source ID to the front
		tempids = [xtemp, ytemp] + tempids 	# adding original coordinates to the front
		if not shorten_df: tempids = tempids + [sources[head][i] for head in dfheaderlist] # adding additional header values at end, if requested
		masterlist.append(tempids)		# saving to the full source list

		# Searching each list of sources from each region file to identify overlaps
		for j in range(len(idlist)): # Number of lists (region files) to search through (e.g. for each catalog, search...)
			for k in range(len(xlist[j])): # Number of sources to search through for the current list/region file (e.g. for each source in a specific catalog...)
				# When overlap is found, see if masterlist has room to add it. 
				# If not, add a new row to make room.
				if tempxmax > xlist[j][k] > tempxmin and tempymax > ylist[j][k] > tempymin and \
				sqrt((xlist[j][k]-xtemp)**2 + (ylist[j][k]-ytemp)**2) <= search_radius[i]: # If the catalog source falls within the range of the base source
					try: 
						# With blockend showing how many total items were found prior to the search on this source, 
						# and tempn showing how many counterparts were identified for the current source, 
						# blockend+tempn should identify the index of the current source
						# The following will cycle through all indices from blockend to blockend+tempn 
						# to see where the last open space is
						for n in range(tempn+1):
							if masterlist[blockend+n][j+3] == None: 
								masterlist[blockend+n][j+3] = idlist[j][k]
								break; # After the last open space, break the chain.
							else: pass;
					except: 
						# Exception will be raised once we reach the end of the current list without finding a free space. 
						# Add a new line, if that's the case.
						tempids = [None]*(len(idlist) + 1) # keeps track of the ids associated with the identified source
						tempids[0] = tempid	# adding current source id to front of list
						tempids = [xtemp, ytemp] + tempids	# adding current coordinates to list
						if not shorten_df: 
							tempids = tempids + [sources[head][i] for head in dfheaderlist] # adding additional header values at end
						tempids[j+3] = idlist[j][k] # Add the source to the list of matches
						masterlist.append(tempids)

					tempn = tempn + 1 # adds a count to the identified sources for this file.

		blockend = len(masterlist) # the index of the end of the previous "block" of sources already detected.

	if verbose: print("DONE WITH CLEANING. CREATING DATAFRAME...")

	# If catalogs not given, use the name of the region files as the headers of the DataFrame
	if not catalogs:
		catalogs = []
		try: 
			for r in regions: catalogs.append(r.split(".reg")[0]) 
		except: 
			for i in len(range(xlist)): catalogs.append("ID "+str(i))
	else: catalogs = [cat+" ID" for cat in catalogs]

	# Adding catalogs to the headers to be read into the DataFrame
	headlist = [coordheads[0], coordheads[1], sourceid]
	for i in catalogs: headlist.append(i)	# adding the name of the catalogs to the header list
	if not shorten_df: headlist = headlist + dfheaderlist	# if requesting, readding other original headers to end of list

	vallist = []
	# Converting the masterlist into an array to be read in as DataFrame values
	temp_array = np.array(masterlist).T
	for i in range(len(temp_array)): vallist.append(temp_array[i].tolist())

	Matches = BuildFrame(headers=headlist, values=vallist)
	Matches.to_csv(outfile)

	return Matches
    
class XrayBinary:
    def __init__(self, df):
        self.df = df.copy()
        self.RA = df['RA'].values
        self.Dec = df['Dec'].values
        if 'CSC ID' in self.df.columns: 
            self.cscid = self.df['CSC ID'].values
        if 'X' in self.df.columns: self.x = self.df['X'].values
        if 'Y' in self.df.columns: self.y = self.df['Y'].values  

    def crossref(self, clusters, catalogs, search_radius=3, sourceid='CSC ID',
                 outfile='xrb_to_cluster_crossref.txt'):
        '''A method to crosreference between the X-ray Binaries and clusters. 
        
        This method will first `Crossref` between the X-ray Binaries and clusters
        to study the ejection of X-ray Binaries from star clusters and add the RA and Dec
        coordinates of those crossreferences to the current dataframe. 

        Parameters:
        clusters : ds9 regions files, list
            DS9 regions files of the clusters to crossreference between
            the X-ray Binaries and the clusters. The coordinates must in fk5.
        catalogs : list 
            Name of the catalogs associated with the region files.
        
        '''

        # Make sure that the cluster regions and the list of their names are of the same size
        # assert(len(clusters) == len(catalogs))

        crossref_df = Crossref(
            df=self.df,
            regions=clusters,
            catalogs=catalogs,
            sourceid=sourceid,
            search_radius=search_radius,
            coordsys='fk5',
            coordheads=['RA', 'Dec'],
            outfile=outfile
        )

        # for cluster, catalog in zip(clusters, catalogs):
        #     ra, dec = GetCoords(infile=cluster)
        #     crossref_df[f'{catalog} RA'], crossref_df[f'{catalog} Dec'] = ra, dec

        return crossref_df

        def _get_coords(df, clusterdf):
            temp = comparedfs(df, clusterdf,
                              )