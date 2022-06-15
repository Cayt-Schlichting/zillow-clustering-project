import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from env import host, username, password


##### DATA ACQUISITION ######
### get_db_url(db_name)
### getNewZillowData()
### getZillowData()
### get_iqr_outlier_bounds(df,include=None,exclude=None)
### def trim_iqr_outliers(df,bounds)
### def handle_iqr_outliers(df,trim=True,include=None,exclude=None)

#Function to create database url.  Requires local env.py with host, username and password. 
# No function help text provided as we don't want the user to access it and display their password on the screen
def get_db_url(db_name,user=username,password=password,host=host):
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

#Function to get new data from Codeup server
def getNewZillowData():
    """
    Retrieves zillow dataset from Codeup DB and stores a local csv file
    Returns: Pandas dataframe
    """
    db_name= 'zillow'
    filename='zillow.csv'
    sql = """
    SELECT *
    FROM properties_2017
        JOIN predictions_2017 USING(parcelid)
        LEFT JOIN airconditioningtype USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE transactiondate LIKE '2017%%' 
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL;
    """
    #Read SQL from file
    df = pd.read_sql(sql,get_db_url(db_name))
    #Drop ID columns
    df.drop(columns=['id'],inplace=True)
    #write to disk - writes index as col 0:
    df.to_csv(filename)
    return df

#Function to get data from local file or Codeup server 
def getZillowData():
    """
    Retrieves Zillow dataset from working directory or Codeup DB. Stores a local copy if one did not exist.
    Returns: Pandas dataframe of zillow data
    """
    #Set filename
    filename = 'zillow.csv'

    if os.path.isfile(filename): #check if file exists in WD
        #grab data, set first column as index
        return pd.read_csv(filename,index_col=[0])
    else: #Get data from SQL db
        df = getNewZillowData()
    return df

##################################
##################################



##### DATA PREPARATION ######
### count_nulls(df,by_column=True)
### handle_missing_values(df, prop_req_col=.75, prop_req_row=.75)
###

def count_nulls(df,by_column=True):
    ''''
    Finds number and percentage of nulls by column or by row

    Returns: Dataframe with null count per row/column
    Parameters:
            df: Dataframe to count the nulls in
      by_column: If True, counts the nulls per column, else counts the nulls per row. Default: True
    '''
    #intialize df
    df_nulls = pd.DataFrame()
    
    if by_column:
        #total rows
        num_rows = df.shape[0]
        #loop over each column
        for c in df.columns:
            #count # of nulls
            null_cnt = df[c].isna().sum()
            #calculate percent of nulls
            null_perc = null_cnt/num_rows
            #Populate DF
            df_nulls.loc[c,'num_nulls'] = null_cnt
            df_nulls.loc[c,'pct_nulls'] = null_perc
    else: 
        #total cols
        num_cols = df.shape[1]
        #loop over each row
        for i in df.index:
            #count # of nulls
            null_cnt = df.loc[i,:].isna().sum()
            #calculate percent of nulls
            null_perc = null_cnt/num_cols
            #Populate DF
            df_nulls.loc[i,'num_nulls'] = null_cnt
            df_nulls.loc[i,'pct_nulls'] = null_perc
    
    return df_nulls

def handle_missing_values(df, prop_req_col=.75, prop_req_row=.75):
    """
    Checks the rows and columns for missing values.  Drops any rows or columns less than the specified percentage.
    
    Returnss: Dataframe
    Parameters: 
                df: Dataframe to analyze
      prop_req_col: Proportion (between 0 and 1) of values in column that must exist (non-null). Default: .75
      prop_req_row: Proportion (between 0 and 1) of values in a row that must exist (non-null). Default: .75
    """
    #Determine number of required non-nulls in columns (prop * num_rows)
    col_thresh = int(round(prop_req_col*df.shape[0],0))
    #drop columns w/o enough info
    df.dropna(axis=1,thresh=col_thresh,inplace=True)
    
    #NOW DO ROWS - note, this on already trimmed df
    #Determine number of required non-nulls in rows (prop * num_cols)
    row_thresh = int(round(prop_req_row*df.shape[1],0))
    #drop rows w/o enough info
    df.dropna(axis=0,thresh=row_thresh,inplace=True)
    
    return df

def get_iqr_outlier_bounds(df,include=None,exclude=None):
    """
    Returns dataframe with list of columns and the upper and lower bounds using the IQR method:
        LB = Q1 - 1.5 * IQR
        UB = Q3 + 1.5 * IQR
    If no columns passed to include or exclude, it defaults to finding outliers for all columns.
    Function will ignore non-numeric columns. FUTURE: Columns that contain only 0s and 1s.
    
    Returns: Pandas Dataframe 
    Parameters:
           df: dataframe in which to find outliers
      include: list of columns to find outliers for
       excude: list of columns to NOT find outliers for.  Ignored if 'include'is set.   
    
    C88
    """
    #Get Column List
    #if include and exclude are None
    if not include and not exclude:
        columns = df.columns #returns index - iterable
    elif include:
        columns = include
    else: columns = exclude
    print(columns)
    
    #Only pull out numeric columns
    columns = df[columns].select_dtypes(include='number')
#     #TO DO: check if only 0s and 1s
#     for c in columns:
#         #if series contains only zeros and ones
#         if df[c].isin([0,1]).all():
    
    #create df for bounds
    bounds = pd.DataFrame()
    #for each column, 
    for col in columns:
        #find bounds
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3 - q1
        lb = q1 - (1.5 * iqr)
        ub = q3 + (1.5 * iqr)
        #put info in df
        bounds.loc['lb',col] = lb
        bounds.loc['ub',col] = ub
    print(bounds)
    return bounds

def trim_iqr_outliers(df,bounds):
    """
    Takes in the dataset dataframe, and dataframe of the bounds for each column to be trimmed.
    Returns: Trimmed dataframe
    """
    #loop over columns to work on
    for col in bounds.col:
        #for each col, grab the outliers
        lb = bounds.loc['lb',col]
        ub = bounds.loc['ub',col]
        #create smaller df of only rows where column is in bounds
        df = df[(df[col] >= lb) & (df[col]<=ub)]
    return df

def handle_iqr_outliers(df,trim=False,include=None,exclude=None):
    """
    Takes in a dataframe and either trims outliers or creates column identifying outliers. 
    
    Outputs: None
    Returns: Pandas Dataframe
    Parameters:
                   df: dataframe in which to find outliers
                 trim: If True, will trim out any rows that contain any outliers.  
                        If False, creates new columns to indicate if row is an outlier or not.
                        Default: False
      include/exclude: list of columns to include or exclude for this function.  
                       Default is all, exclude will be ignored if include is provided.
    """    
    #Get bounds dataframe
    bounds = get_iqr_outlier_bounds(df,include,exclude)

    #If we want new column 
    if not trim:
        #Function trims row if value not w/i bounds
        df = trim_iqr_outliers(df,bounds)
    else:
        #function adds new columns 
        # for each column, adds value corresponding to if in or out of bounds
        #NOT SURE HOW DO DO THIS YET.
        # For each column:
        # add new column_outliers
        # func(x, bounds)
        # args that are innate
        x =2
    return df

##################################
##################################