import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from env import host, username, password

#helper functions
import utils


##### DATA ACQUISITION ######
### get_db_url(db_name)
### getNewZillowData()
### getZillowData()

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
# #################################

#############################
# #### DATA PREPARATION ######

def only_single_unit(df):
    '''
    This function takes in a zillow dataframe and drops any rows with unitcnt > 2(keeps nulls).
    It also keeps parcels of the following property land use type:  
        Single Family (261), PUD (269), Mobile home (263), Townhouse (264), Condominium (266),
        Manufactured, etc (275), Residential general (260), Rural residence (262), Bungalow (273),
        Zero Lot Line (274), Inferred Single Family (279), Patio Home (276)
    '''
    #only keep rows with unit count of 1 or null
    df = df[(df.unitcnt ==1) | df.unitcnt.isna()]
    #filter land use type
    df = df[df.propertylandusetypeid.isin([261,269,263,264,266,275,260,262,273,274,279,276])]
    return df

def map_encode_zillow_fips(df):
    #map to county names
    df['county'] = df.fips.map({6037: 'LosAngeles',6059:'Orange',6111:'Ventura'})
    #encode into dummy df
    d_df = pd.get_dummies(df['county'],drop_first=True)
    #concat dummy df to the rest
    df = pd.concat([df,d_df],axis=1)
    #Drop fips
    df.drop(columns=['fips'],inplace=True)
    return df

def splitData(df,**kwargs):
    """
    Splits data into three dataframes
    Returns: 3 dataframes in order of train, test, validate
    Inputs:
      (R)             df: Pandas dataframe to be split
      (O -kw)  val_ratio: Proportion of the whole dataset wanted for the validation subset (b/w 0 and 1). Default .2 (20%)
      (O -kw) test_ratio: Proportion of the whole dataset wanted for the test subset (b/w 0 and 1). Default .1 (10%)
    """
    #Pull keyword arguments and set test and validation percentages of WHOLE dataset 
    val_per = kwargs.get('val_ratio',.2)
    test_per = kwargs.get('test_ratio',.1)

    #Calculate percentage we need of test/train subset
    tt_per = test_per/(1-val_per)

    #Split validate dataset off
    #returns train then test, so test_size is the second set it returns
    tt, validate = train_test_split(df, test_size=val_per,random_state=88)
    #now split tt in train and test 
    train, test = train_test_split(tt, test_size=tt_per, random_state=88)
    
    return train, test, validate

def prep_zillow(df):
    '''
    This function takes in a zillow dataframe and prepares it for exploratory analysis.
    '''
    #FILTER down to 'single unit' properties
    df = only_single_unit(df)
    
    #NULLS: filter down null-heavy columns, then drop rows with nulls
    df = utils.handle_missing_values(df,prop_req_col=.98,prop_req_row=1)
    
    #DROP remaining columns I don't want - do geographic columns separately to easily change
    drp_cols = ['propertylandusetypeid', 'finishedsquarefeet12', 'structuretaxvaluedollarcnt','calculatedbathnbr',\
                'assessmentyear','landtaxvaluedollarcnt', 'taxamount','propertycountylandusecode']
    drp_geo_cols = ['regionidcity','regionidcounty', 'regionidzip','rawcensustractandblock','censustractandblock']
    df.drop(columns=(drp_cols+drp_geo_cols), inplace=True)
#     df.drop(columns=drp_geo_cols, inplace=True)

    #CONVERT datatypes
    to_int_cols = ['bedroomcnt', 'yearbuilt', 'fullbathcnt', 'latitude', 'longitude','roomcnt','calculatedfinishedsquarefeet']
    df[to_int_cols] = df[to_int_cols].astype(int)
    
    #CONVERT transactiondate
    df['trans_month'] = pd.DatetimeIndex(df['transactiondate']).month
    df.drop(columns='transactiondate',inplace=True)
    
    #MOVE parcelid to index
    df.set_index('parcelid',drop=True,inplace=True)
    
    #MAP and ENCODE fips
    df = map_encode_zillow_fips(df)

    #ENCODE landusedesc
    #encode into dummy df
    d_df = pd.get_dummies(df['propertylandusedesc'],drop_first=True)
    #concat dummy df to the rest
    df = pd.concat([df,d_df],axis=1)

    #TRIM IQR outliers
    iqr_trim_cols = ['bedroomcnt','bathroomcnt','calculatedfinishedsquarefeet', 'taxvaluedollarcnt','logerror']
    df = utils.handle_iqr_outliers(df,trim=True,include=iqr_trim_cols)
    
    #RENAME columns
    rename_cols = {
        'bedroomcnt':'bed',
        'bathroomcnt':'bath',
        'calculatedfinishedsquarefeet':'sf',
        'latitude': 'lat',
        'longitude':'lon',
        'propertylandusedesc':'landusedesc',
        'taxvaluedollarcnt':'value'
    }
    df.rename(columns=rename_cols,inplace=True)
    
    return df    

def wrangle_zillow(**kwargs):
    '''
    Acquires, preps and splits zillow data.  Can pass in val_ratio or test_ratio for splitting
    '''
    #acquire data
    df = getZillowData()
    #prep data
    df = prep_zillow(df)
    #split data
    train, test, validate = splitData(df,**kwargs)
    
    return train, test, validate    
##################################
# #################################
