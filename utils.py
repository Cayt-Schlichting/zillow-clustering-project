import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

###### SCALE FUNCTION ######
def scale_X(X_tr,X_te,X_val,columns,kind='minmax'):
    '''
    Takes prepped tr, test, validate zillow subsets. Scales the list of columns provided\
      returns dataframes concated <scaled><unscaled>.  
      
    Returns: 3 Pandas DataFrames (Train, Test, Validate)
    Inputs:
         (R) X_tr: train dataset
         (R) X_te: test dataset
        (R) X_val: validate dataset
      (R) columns: List of columns to be scaled
      (O-kw) kind: Type of scaler you want to use.  Default: minmax
                Options: minmax, standard, robust
    '''
    #Set the scaler 
    if kind.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif kind.lower() == 'standard':
        scaler = StandardScaler()
    elif kind.lower() == 'robust':
        scaler = RobustScaler()
    else:
        print(f'Invalid entry for "kind", default MinMax scaler used')
        scaler = MinMaxScaler()

    #fit scaler and transform on train - needs to be stored as pd.DF in order to concat
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr[columns]),columns=columns,index=X_tr.index)
    #transform the rest
    X_te_scaled = pd.DataFrame(scaler.transform(X_te[columns]),columns=columns,index=X_te.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val[columns]),columns=columns,index=X_val.index)

    #rebuild the dataframes <scaled><unscaled>
    X_tr_scaled = pd.concat([X_tr_scaled,X_tr.drop(columns=columns)],axis=1)
    X_te_scaled = pd.concat([X_te_scaled,X_te.drop(columns=columns)],axis=1)
    X_val_scaled = pd.concat([X_val_scaled,X_val.drop(columns=columns)],axis=1)
    
    #return dataframes with scaled data
    return X_tr_scaled, X_te_scaled, X_val_scaled

##################################
##################################

###### NULL FUNCTIONS #####
### count_nulls(df,by_column=True)
### handle_missing_values(df, prop_req_col=.75, prop_req_row=.75)

# Improvements to make
# 1) needs to be restructured to better handle lots of rows or columns (takes to long to run)
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

##################################
# #################################



###### OUTLIER FUNCTIONS #####
### get_iqr_outlier_bounds(df,include=None,exclude=None)
### trim_iqr_outliers(df,bounds)
### calc_outliers(x,lb,ub)
### add_outlier_columns(df,bounds)
### handle_iqr_outliers(df,trim=True,include=None,exclude=None)

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
    return bounds

def trim_iqr_outliers(df,bounds):
    """
    Takes in the dataset dataframe, and dataframe of the bounds for each column to be trimmed.
    Returns: Trimmed dataframe
    """
    #loop over columns to work on
    for col in bounds.columns:
        #for each col, grab the outliers
        lb = bounds.loc['lb',col]
        ub = bounds.loc['ub',col]
        #create smaller df of only rows where column is in bounds
        df = df[(df[col] >= lb) & (df[col]<=ub)]
    return df

def calc_outliers(x,lb,ub):
    ''''
    Given a value, determines if it is between the provided upper and lower bounds.  
    If b/w bounds, returns 0, else returns the distance outside of the bounds
    '''
    #if not an outlier, set to zero
    if lb <= x <= ub: return 0
    elif x < lb: return x-lb
    else: return x-ub

def add_outlier_columns(df,bounds):
    #loop over columns in bounds
    for col in bounds.columns:
        #new column name
        col_name = col + '_outlier'
        #for each column, apply the outlier calculation and store to new column
        df[col_name] = df[col].apply(calc_outliers,args=(bounds.loc['lb',col],bounds.loc['ub',col]))
    return df

##TIME PERMITTING
# Investigate why python is updating my dataframe external to the function
# needed to use df = old_df.copy() to prevent this
def handle_iqr_outliers(old_df,trim=False,include=None,exclude=None):
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
    df= old_df.copy()    
    #Get bounds dataframe
    bounds = get_iqr_outlier_bounds(df,include,exclude)

    #If we want new column 
    if trim:
        #Function trims row if value not w/i bounds
        df = trim_iqr_outliers(df,bounds)
    else:
        #function determines if outlier and adds new columnadds new columns 
        df = add_outlier_columns(df,bounds)
        
    return df
##################################
# #################################
