import numpy as np
import pandas as pd

def make_circular_regions(radius):
    reg = "circle("
    reg_props = [f", {r}) #" for r in radius]
    return reg, reg_props

def make_point_regions(marker, radius):    
    reg = "point("  
    reg_props = [f") # point={marker} {r}" for r in radius]
    return reg, reg_props

def make_ruler_regions(df, coordsys, radunit):
    reg = "# ruler("
    reg_props = [f") ruler = {coordsys} {radunit}" for i in range(len(df))] # Not sure what vector=1 is doing
    return reg, reg_props

def make_vector_regions(x0, y0, x1, y1):
    reg = "# vector("
    reg_props = [") vector = 1" for i in range(len(x0))]
    x0, y0, x1, y1 = np.array(x0), np.array(y0), np.array(x1), np.array(y1)
    length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) # distance formula
    theta = np.arctan2(y1, (x1-x0)) # tan theta = opposite / adjacent
    return reg, reg_props, length, theta
    
def WriteReg(sources, outfile, coordsys=False, coordheads=False, reg_type='circle', additional_coords=False, coordnames=False, idheader=False, color="#FFC107", radius=False, radunit=False, label=False, width=1, fontsize=10, bold=False, dash=False, addshift=[0,0], savecoords=None, marker=False, fill=False): 

    """
    Writes a DS9 region file for all sources given a DataFrame containing their coordinates or a list of the source coordinates. 
    
    PARAMETERS
    ----------
    sources     [DataFrame, list]:	Sources for which to plot the regions. Can be provided as either a 
                                        DataFrame containing the the galactic or image coordinates, or a list of 
                                        coordinates in [[xcoords], [ycoords]] format.
    outfile     [str]		:	Name of the file to save the regions to.
    coordsys    [str]		:	Defines the coordinate system to use in DS9. Options are 'image' or 'fk5'.
					If no coordsys is given, will attempt to infer from other inputs.
    coordheads  [list]		:	Name of the headers containing the coordinates of each source, 
                                        read in as a list in [xcoordname, ycoordname] format. This is only needed if
					sources is a DataFrame. If coordheads is not defined, will attempt to infer
					the correct coordinate headers from the DataFrame or other inputs.
    reg_type   [str]        : The type of regions to create. The current options are `circle`, `point`, and `ruler`.
    additional_coords   [list] : If `reg_type='ruler'`, will use these coordinates for creating ruler regions. These
                    should be the coordinates of the source the ruler points to. 
    coordnames  [list]		: 	Depricated parameter, now called coordheads (as of v1.6.0).
    idheader    [str]           :   	Name of the header containing the ID of each source. By default, checks 
                                        whether the DataFrame contains a header called 'ID'. If not, it's assumed
                                        no IDs are given for each source. 
    idname 	[str] 		: 	Depricated parameter, now called idheader (as of v1.6.0)
    color 	[str] 		: 	Color of the regions to plot. Default is '#FFC107' (yellow-orange).
    radius 	[int] 		: 	Radius or size of the region to plot, given as a list of unique values or a 
					single value to use on all sources. If no radius is given, all radii are set to 
					10 pixels or 1 arcsecond, depending on radunits or the coordinate system.
    radunit 	[str] 		: 	Unit of the region radius. If no unit is given, algorithm makes a guess at the
					best unit to use based on the coordinate system. 
    label 	[list] 		: 	Labels to apply to each source. This is overwritten if idheader is given.
    width 	[int] 		: 	Width of the region outline if circular regions are used.
    fontsize 	[int]		:	Size of the label text. 
    bold 	[bool] 		: 	If True, sets text to boldface. Default is False.
    dash 	[bool] 		:	If True, sets region circles to be outlined with dashed lines. Default is False.
    addshift 	[list] 		: 	Adds a shift to the source coordinates. Shifts must be given in the same units as the coordinates!
    savecoords 	[str] 		: 	Saves the source coordinates to a text file with the input filename. 
    marker 	[str] 		: 	Defines a marker to use instead of circular units. For DS9, options include circle, box, diamond, 
					cross, x, arrow, or boxcircle.
    fill 	[bool] 		: 	If True, fills the region with the color determined by the 'color' parameter. 
					Only circular regions can be filled. 
    """

    # Coordnames has been renamed coordheads to standardize parameter names across XRBID, 
    # This code is added so that old calls of WriteReg still work without error. 
    if coordnames: coordheads=coordnames

    # xcoord and ycoord keep track of the coordinate headers if source is a DataFrame
    xcoord = False
    ycoord = False
    
    # Pulling the name of the coordinate headers for source DataFrame, if given
    if coordheads: 
        xcoord = coordheads[0]
        ycoord = coordheads[1]

    # Pulling source coordinates from sources, depending on the type of input
    if isinstance(sources, list): # if sources is a list, assume they are a list of coordinates
        if len(np.asarray(sources)) == 2: # if in the format [x_coords, y_coords]....
            x_coords = sources[0]
            y_coords = sources[1]
        else: # if not, assume given as [[x1,y1], [x2,y2]...]
            x_coords = np.array(sources).T[0]
            y_coords = np.array(sources).T[1]
    elif isinstance(sources, str) and len(re.split(r"\.", sources)) > 1: # if sources is a filename, use GetCoords to retrieve coordinates
        x_coords, y_coords = GetCoords(infile=sources)
    elif isinstance(sources, pd.DataFrame): # If df read in, try to decide on coordinate system, if not given
        if not coordheads:
            # The header name can be inferred from the coordinate system
            # Assumes image coordinates have headers [X,Y] or [x,y] and 
            # fk5 coordinates have headers [RA,Dec] or [ra,dec]
            if not coordsys or coordsys.lower() == "fk5": 
                if "RA" in sources.columns.tolist():
                    xcoord = "RA"
                elif "ra" in sources.columns.tolist(): 
                    xcoord = "ra" 
                else: pass;
                if "Dec" in sources.columns.tolist(): 
                    ycoord = "Dec" 
                elif "dec" in sources.columns.tolist(): 
                    ycoord = "dec"
                else: pass;
            if not coordsys or "im" in coordsys.lower(): 
                if "X" in sources.columns.tolist(): 
                    xcoord = "X" 
                elif "x" in sources.columns.tolist(): 
                    xcoord = "x" 
                else: pass; 
                if "Y" in sources.columns.tolist(): 
                    ycoord = "Y" 
                elif "y" in sources.columns.tolist():
                    ycoord = "y" 
                else: pass;                
        # Whether or not coordheads is given, we should have a valid xcoord and ycoord by now
        # If not, this code will pass an error, and the user will be asked to provide the header names. 
        try: 
            x_coords = sources[xcoord].values.tolist()
            y_coords = sources[ycoord].values.tolist()
        except: 
            coordheads = input("Coordinate headers not found. Please enter headers (separated by comma, no space):")
            xcoord,ycoord = coordheads.split(",")
            # If this still fails, then user has erred and program will end in error. 
            x_coords = sources[xcoord].values.tolist()
            y_coords = sources[ycoord].values.tolist()

    # At this point, WriteReg should now have the x and y coordinates of each region to plot. 
    # We also need to make sure coordsys, radius, and radunits are known, if not given. 

    # Sorting out coordsys options
    if coordsys: 
        coordsys = coordsys.lower()
        if "im" in coordsys: coordsys = "image" # allows user to input img instead of image and still get same results
    if not coordsys: 
        # If any values are beyond the acceptable range of fk5, this must be in image coordinates
        if max(x_coords) > 360 or max(np.abs(y_coords)) > 90 or xcoord in ["X","x"]: coordsys = "image"
        else: coordsys="fk5"

    # Setting up the units to add to the radii
    if (radunit and 'arcs' in radunit) or (not radunit and coordsys=="fk5"): radmark='\"' # add arcsec
    elif (radunit and 'arcm' in radunit): radmark='\'' # add arcmin
    elif (radunit and 'deg' in radunit): radmark='d'   # add degree marker
    else: radmark=''  # if pixels or marker is given, no unit marker needed
        
    # Setting default radius, if not given
    if isinstance(radius, list): radius = [str(r)+radmark for r in radius] # converting radii to strings with unit markers added
    elif not radius: # defaults to 3 pixels or 0.5 arcsec, depending on coordinate system and marker type
        if coordsys == "image" or marker: radius = ['10']*len(x_coords) # pixels
        else: radius = ['1'+radmark]*len(x_coords) # will add unit arcsecond soon
    elif not isinstance(radius, list): radius = [str(radius)+radmark]*len(x_coords) # making radius a list of strings, if single value given 

    # If only one width is given, use it as the default in the header
    # otherwise, unique widths will be added to each region, and the header width will be set to the first in list
    if not isinstance(width, list): 
        uniquewidth = False
        defaultwidth=width
    else:
        uniquewidth = True
        defaultwidth = width[0]
        
    # Now have radii as a list of strings with the unit markers included, 
    # coordinates of each region to plot, and the coordinate system 
    # Can start to put together the strings to write to file.
    
    #### PARAMETERS FOR PLOTTING #####        
    # Setting text to bold or normal based on user input
    if bold: bold = " bold"
    else: bold = " normal" 
    # Setting up the header of the region file based on parameters read in
    f_head = "# Region file format: DS9 version 4.1\nglobal color=" +str(color)+" dashlist=8 3 width="+str(defaultwidth)+\
            " font=\"helvetica "+str(fontsize)+bold+" roman\" select=1 highlite=1 dash="+str(int(dash))+\
            " fill="+str(int(fill))+" fixed=0 edit=1 move=0 delete=1 include=1 source=1\n"+coordsys+"\n" 
    # NOTE, each source can theoretically have a different width, but if not, we'll want to set the width in the header of the .reg

    # SETTING UP REGION OF EACH SOURCE
    # reg determines the type of region (circle or point), preceeding the coordinates of each point
    # reg_props determines the properties (label, color, etc) of each point following the coordindates of each point
    # If it is a marker, radius is the markersize and added to reg_props after the parenthesis        
    if reg_type.lower() == 'point': 
        reg, reg_props = make_point_regions(marker, radius)
    elif reg_type.lower() == 'circle': # If the region is a circle, the radius is added within the parenthesis
        reg, reg_props = make_circular_regions(radius)
    elif reg_type.lower() == 'ruler':
        x1_coords = sources[additional_coords[0]]
        y1_coords = sources[additional_coords[1]]
        reg, reg_props = make_ruler_regions(sources, coordsys, radunit)
    elif reg_type.lower() == 'vector':
        x1_coords = sources[additional_coords[0]]
        y1_coords = sources[additional_coords[1]]
        reg, reg_props, length, theta = make_vector_regions(x_coords, y_coords, 
                                                            x1_coords, y1_coords)
    else:
        raise ValueError(f"{reg_type} invalid region type.")
        # ValueError.add_note(f"{reg_type} invalid region type.")
    # If list of labels or idheader given, add them to the end of reg_props
    if idheader: label = sources[idheader].values.tolist()
    if label: reg_props = [r+" text={"+str(label[i])+"}" for i,r in enumerate(reg_props)]
    # If each source has a unique width, add them to reg_props
    if uniquewidth: reg_props = [r+" width="+str(width[i]) for i,r in enumerate(reg_props)]

    # Finally, adding endline character to end of reg_props to place each region on a new line in the .reg file
    reg_props = [r+"\n" for r in reg_props] 
    
    # Putting together the full region line using the coordinates of each source
    if reg_type == 'ruler':
        f_reg = [f"{reg}{x_coords[i] + addshift[0]}, {y_coords[i] + addshift[1]}, {x1_coords[i] + addshift[0]}, {y1_coords[i] + addshift[1]}{r}" for i, r in enumerate(reg_props)]
    elif reg_type == 'vector':
        f_reg = [f"{reg}{x_coords[i] + addshift[0]}, {y_coords[i] + addshift[1]}, {length[i]}\", {theta[i]}{r}" for i, r in enumerate(reg_props)]
    else:
        f_reg = [f"{reg}{x_coords[i] + addshift[0]}, {y_coords[i] + addshift[1]}{r}" for i, r in enumerate(reg_props)]
        # f_reg = [reg+str(x_coords[i]+addshift[0])+", "+str(y_coords[i]+addshift[1])+r for i,r in enumerate(reg_props)]

    #### WRITING THE REGION FILE ####
    print("Saving", outfile)
    with open(outfile, 'w') as f:
        f.write(f_head)
        for r in f_reg: # for each source in the list, print the 
            f.write(r)
    print(outfile,"saved!")

    if savecoords: 
        with open(savecoords, "w") as f: 
            np.savetxt(f, np.column_stack([x_coords, y_coords]))
        print(savecoords, "saved!")

# if __name__ == "__main__":

    