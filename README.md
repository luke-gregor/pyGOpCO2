# UnderWay pCO<sub>2</sub>

This repo contains scripts that process underway pCO<sub>2</sub> data from raw data files (*dat.txt).

#### Level 0 Data
This data is strictly raw data. It also contains the error files. 

#### Level 1 Data

**Level 1a:** data is only the concatenated data.  [`level0_to_level1.py`](level0_to_level1.py) concatenates all files in a given folder. The concatenated data is stored in a single file. Lines with `ERROR != 0` are discarded.  
**Level 1b:** This also offers an opportunity to fill in missing data. This includes Lats and Lon (for the early Agulhas II cruises) and P<sup>atm</sup> where it is required. The output from this data is stored as level **1b** data. If all data is present then level 1a and 1b will look identical. There may need to be additional processing to fix latitiudes and longitudes for more recent files where Sea Data System (SDS) data was streamed in wirelessly. 

#### Level 2 Data
This takes input from level **1b**. Conversion from xCO2 to fCO2 and then quality control takes place in this step. This step replaces the Pierrot et al. 2009 MATLAB script. This has been compared with MATLAB output. It matches almost identically. See figure below.![title](../test/GO-underwayCO2_MATLAB-Python_compare.png)

## fCO<sub>2</sub> calculation workflow

1. Concatenate files
2. Select subset of data using df.ix['yyyy-mm-dd':]
3. Calculate fCO2 from all other data (might need to fix salinity dependencies)
4. Plot QC diagnostics plot and see if any limits need adjustment
5. If limits need adjustment do so using level1_to_level2.flag4.update()
6. Run QC that creates flags (2 | 4)
7. Create a new DataFrame that contains only the QC'd data
8. Create UnderdwayCO2(df) to call EQU, ATM or STDx data
9. Plot oceanography parameters

## INFO

**Author:**  Luke Gregor  
**Contact:** [luke.gregor@outlook.com](mailto:luke.gregor@outlook.com)  
**Date:**   2017-10-04  
**Version:** 0.1


## TODO!
#### High priority
- Create a script for 1a to 1b that tests for missing data (coordinates, temperatures, pressures, xCO2) – informs the user. If everything is present then creates 1b data  

#### Low priority
- find and download NCAR data for missing Patm (for level 1a-1b) OpenDAP?
- Fetch wind data for flux calculations - I have a script for (fCO2 + wind + temp + salt) to FCO2 - Maybe using OpenDAP