# Investigating Zestimate Errors
*Audience: Target audience for my final report is a data science team*


<hr style="background-color:silver;height:3px;" />

## Project Summary
<hr style="background-color:silver;height:3px;" />

The goal of this project is to identify drivers of error in zestimate.  Understanding when the zestimate is likely to be less reliable will help us to develop better models for those situations.

### Project Deliverables
> - A final report notebook
> - Python modules for automation and to facilitate project reproduction
> - Notebooks that show:
>  - Data acquisition and preparation 
>  - exploratory analysis not included in final report
>  - model creation, refinement and evaluation

### Initial questions on the data

>  - Questions
>  - Thoughts
>  - etc

### Project Plan 

- [ ] **Acquire** data from the Codeup SQL Database. 
- [ ] Clean and **prepare** data for the explore phase. 
- [ ] Create wrangle.py to store functions I created to automate the cleaning and preparation process. 
- [ ] Separate train, validate, test subsets and scaled data.
- [ ] **Explore** the data through visualization and hypothesis testing.
    - [ ] Clearly define at hypotheses and questions.
    - [ ] Document findings and takeaways.
- [ ] Perform **modeling**:
   - [ ] Identify model evaluation criteria
   - [ ] Create at least three different models.
   - [ ] Evaluate models on appropriate data subsets.
- [ ] Create **Final Report** notebook with a curtailed version of the above steps.
- [ ] Create and review README. Ensure it contaions:
   - [ ] Data dictionary
   - [ ] Project summary and goals
   - [ ] Initial Hypothesis
   - [ ] Executive Summary
---

<hr style="background-color:silver;height:3px;" />

## Executive Summary
<hr style="background-color:silver;height:3px;" />

**Project Goal:**

**Discoveries and Recommendations**


<hr style="background-color:silver;height:3px;" />

## Data Dictionary
<hr style="background-color:silver;height:3px;" />

|Target|Definition|
|:-------|:----------|
| logerror | log(zestimate) - log(actual value)|

|Feature|Definition|
|:-------|:----------|
| bedroomcnt (bed)       | Number of bedrooms in the home |
| bathroomcnt (bath)        | Number of bathrooms in home, including fractional bathrooms |
| calculatedfinishedsquarefeet (sf)|  Calculated total finished living area of the home  |
| fips (county)|  Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details |
| fullbathcnt | Number of full bathrooms (sink + shower + bathtub + toilet) |
| latitude        |  Latitude of the middle of the parcel multiplied by 10e6 |
| longitude       |  Longitude of the middle of the parcel multiplied by 10e6 |
| parcelid        | Unique identifier for parcels |
| propertylandusetypedesc|  Type of land use the property is zoned for |
| roomcnt |  Total number of rooms in the principal residence |
| unitcnt |  Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...) |
| yearbuilt |  The Year the principal residence was built |
|taxvaluedollarcnt|The total tax assessed value of the parcel|
|trans_month|Month of the parcel's transaction|





<hr style="background-color:silver;height:3px;" />

## Reproducing this project
<hr style="background-color:silver;height:3px;" />

> In order to reproduce this project you will need your own environment file and access to the database. You can reproduce this project with the following steps:
> - Read this README
> - Clone the repository or download all files into your working directory
> - Add your environment file to your working directory:
>  - filename should be env.py
>  - contains variables: username, password, host
> - Run the Final_Report notebook or explore the other notebooks for greater insight into the project.

