# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import janitor
import sklearn as skl
import random
import seaborn as sns

# Clean age related variables


def clean_age(df):
    """ 
    Cleans the age variable ('agetwoplus') by converting numeric flags to 'Yes'/'No'
    and imputes missing values based on 'ageinmonth' (24 months = 2 years).

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame with imputed age values.
    """
    #  Standardize numeric codes to Yes/No strings
    df['agetwoplus'] = df['agetwoplus'].replace(
        {1.0: 'Under 2 Years Old', 2.0: '2 Years Old or Older'})

    #  Identify rows missing 'agetwoplus'
    is_missing = df['agetwoplus'].isna()
    has_months = df['ageinmonth'].notna()

    #  Vectorized Imputation using boolean masks
    under_2_years = is_missing & has_months & (df['ageinmonth'] < 24)
    over_2_years = is_missing & has_months & (df['ageinmonth'] >= 24)

    df.loc[under_2_years, 'agetwoplus'] = 'Under 2 Years Old'
    df.loc[over_2_years, 'agetwoplus'] = '2 Years Old or Older'

    return df

# Clean GCS Scores


def clean_gcs(df, include_sever_cases=False):
    """ 
    This function cleans all gcs vvariables: gcstotal, gcseye, gcsverbal, gcsmotor. We keep the numeric scores for these variables as they are clinically meaningful and we will use them in our analysis.
    It first ensure that all gcstotal are the sum of gcseye, gcserbal, and gcsmotor, if not, then we will impute the total as the sum of the three subvariables. If there are Na in the subvariables then the total will stand as is.
    If there are any missing total variables and all three of the subvariables are populated, then the total will be imputed as the sum of the three subvariables. If any of the subvariables are missing, then the total will remain missing (NaN).
    We then check to ensure gcsgroup is consistent with total score (severe = 3-13 and moderate = 14-15). If there are any inconsistencies, then we will impute the gcsgroup based on the total score. If the total score is missing, then the gcsgroup will remain missing (NaN).
    If include severe cases is True then patients with gcstotal less than 14 will be included, otherwise they will be excluded from the dataset.

    """

    gcs_columns = ['gcstotal', 'gcseye', 'gcsverbal', 'gcsmotor', 'gcsgroup']

    df['gcsgroup'] = df['gcsgroup'].replace({1.0: 'Severe', 2.0: 'Moderate'})

    # Check if gcstotal is the sum of gcseye, gcsverbal, and gcsmotor. If not, then impute the total as the sum of the three subvariables. If there are Na in the subvariables then the total will stand as is.
    df['gcstotal'] = df.apply(
        lambda row: row['gcseye'] + row['gcsverbal'] + row['gcsmotor']
        if pd.notnull(row['gcseye']) and pd.notnull(row['gcsverbal']) and pd.notnull(row['gcsmotor'])
        else row['gcstotal'],
        axis=1
    )

    # Check to ensure gcsgroup is consistent with total score (severe = 3-13 and moderate = 14-15). If there are any inconsistencies, then impute the gcsgroup based on the total score. If the total score is missing, then the gcsgroup will remain missing (NaN).
    df['gcsgroup'] = df.apply(
        lambda row: 'Severe' if row['gcstotal'] < 14 else (
            'Moderate' if row['gcstotal'] >= 14 else row['gcsgroup']),
        axis=1
    )

    # If all subvaribales are NA but GCS total is populated with 15, then we will impute the subvariables as 4, 5, and 6 for eye, verbal, and motor respectively. This is because a GCS total of 15 is the maximum score and indicates that the patient is fully alert and oriented, which corresponds to the maximum scores for each of the subvariables.
    df.loc[(df['gcstotal'] == 15) & (df[['gcseye', 'gcsverbal', 'gcsmotor']].isna(
    ).all(axis=1)), ['gcseye', 'gcsverbal', 'gcsmotor']] = [4, 5, 6]

    # If include severe cases is True then patients with gcstotal less than 14 will be included, otherwise they will be excluded from the dataset.
    if not include_sever_cases:
        df = df[df['gcsgroup'] == 'Moderate']

    return df


# Clean Altered Mental State related variables


def clean_ams(df):
    """
    Cleans the AMS columns in the given DataFrame as well as performs various realibility checks of the data to ensure there are no inconsistencies.
    This includes ensureing all missing values are encoded as NaN, and that all values are either 1.0, 0.0, or NaN. It also checks that for all rows of ams where there is an NaN, that all of the other AMS columns should also be missing the entry for that row. 
    If there are any such rows, then I will impute the value of ams for that row based on the other AMS columns. If any of the other AMS columns are populated, then I will set ams to "Yes" for that row. If all of the other AMS columns are missing, then I will set ams to "No" for that row.
    There are no judgement calls for this variable, any imputations are logical progressions of the data and are not based on any assumptions about the patient or their condition.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing AMS columns.

        Returns: pd.DataFrame: The cleaned DataFrame.


    """

    ams_columns = ['amsagitated', 'amssleep', 'amsslow', 'amsrepeat', 'amsoth']

    # Change all float values of 92.0 in amc Col to NaN
    for col in ams_columns:
        df[col] = df[col].replace(92.0, np.nan)

    # Changing the value of the AMS subvaribales columns to be more interpretable. 1.0 = Yes, 0.0 = No, NaN = Missing
    for col in ams_columns:
        df[col] = df[col].replace((1.0, 0.0), ('Yes', 'No'))
    df['ams'] = df['ams'].replace((1.0, 0.0), ('Yes', 'No'))

    # Impute the value of ams for rows where it is NaN based on the other AMS columns and a GCStotal of less than 15. If any of the other AMS columns are populated, then I will set ams to "Yes" for that row. If all of the other AMS columns are missing, then I will set ams to "No" for that row.
    # We also ensure if there are no;s for the ams columns then ams is set to "No" for that row. This is because if there are no;s for the ams columns then it is logical to assume that the patient does not have AMS and therefore ams should be set to "No" for that row.
    # If all of the other AMS columns are missing, then I will set ams to "No" for that row.

    missing_ams_df = df.loc[df['ams'].isna()].copy()

    for i, row in missing_ams_df.iterrows():
        low_gcs = pd.notna(row['gcstotal']) and row['gcstotal'] < 15
        has_ams_indicator = row[ams_columns].eq('Yes').any()

    for i, row in missing_ams_df.iterrows():
        gcs = row['gcstotal']
        populated_ams = {
            column: row[column]
            for column in ams_columns
            if pd.notnull(row[column])
        }
        low_gcs = pd.notnull(gcs) and gcs < 15

        if low_gcs or populated_ams:
            if low_gcs:
                df.at[i, 'ams'] = 'Yes'
            elif populated_ams:
                df.at[i, 'ams'] = 'Yes'
            else:
                df.at[i, 'ams'] = 'No'

    return df

# Clean Hematoma Realted Variables


def clean_hema(df):
    """
    This function cleans all variable related to Hematoma: hema, hemaloc, and hemasize. All missing values are recorded as NaN.
    For hema, all values are either Yes or No (or NaN) corresponding to 1 and 0 respectivly.
    For hemaloc, all 92 are mapped to NaN and we check to ensure there are no unexpected values not 1, 2, or 3 
    For hema size, we fix 92 to make it NaN additionally we check to ensure there are no unexpected values not 1, 2, or 3
    We then look to see if we can back inpute on the indicator hema, if there is a hema size or location, then we can set hema to Yes. If there is no hema size or location, then we can set hema to No.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing hema columns.


        Returns: pd.DataFrame: The cleaned DataFrame.
    """

    hema_columns = ['hema', 'hemaloc', 'hemasize']

    # Change all float values of 92.0 in hema variables to NaN
    for col in hema_columns:
        df[col] = df[col].replace(92.0, np.nan)

    # Rename the values of the hemavaribales for interpretability.
    df['hema'] = df['hema'].replace((1.0, 0.0), ('Yes', 'No'))
    df['hemaloc'] = df['hemaloc'].replace(
        (1.0, 2.0, 3.0), ('frontal', 'occipital', 'parietal/temporal'))
    df['hemasize'] = df['hemasize'].replace(
        (1.0, 2.0, 3.0), ('small', 'medium', 'large'))

    # Check for any unexpected values in the hema variables; if there is such a value then fill in NaN.
    for col in hema_columns:
        unexpected_values = df[~df[col].isin(
            ['Yes', 'No', 'frontal', 'occipital', 'parietal/temporal', 'small', 'medium', 'large']) & df[col].notnull()]
        if not unexpected_values.empty:
            df.at[unexpected_values.index, col] = np.nan

    # If NaN for the hema variable, then check the other two variables to see if we can back inpute the value of hema. If there is a hema size or location, then we can set hema to Yes. If there is no hema size or location, then we can set hema to No.
    hema_missing_df = df[df['hema'].isna()]
    for i, row in hema_missing_df.iterrows():
        hema_size = row['hemasize']
        hema_loc = row['hemaloc']
        if pd.notnull(hema_size) or pd.notnull(hema_loc):
            df.at[i, 'hema'] = 'Yes'
        else:
            df.at[i, 'hema'] = 'No'

    return df

# Clean Loss of Consciousness Related Variables


def clean_loc(df, suspected_loc):
    """
    This function cleans the two LOC variables: locseparate and loclen.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing LOC columns.
        suspected_loc (Boolean): If True, then the function will treat loc length values corresponding to loc separate = "suspected" as valid inputs with clinical significance
                                If Flase, the the function will treat loc length values corresponding to loc separate = "suspected" as invalid inputs and will set them to NaN.

        Returns: pd.DataFrame: The cleaned DataFrame.
    """

    # Make locseperate and loclen more interpretable by replacing numeric codes with descriptive strings.
    df['locseparate'] = df['locseparate'].replace(
        (0.0, 1.0, 2.0), ('No', 'Yes', 'Suspected'))

    df['loclen'] = df['loclen'].replace((1.0, 2.0, 3.0, 4.0, 92.0), ('less than 5 seconds',
                                        'between 5 second and 1 minute', 'between 1 and 5 minutes', 'greater than 5 minutes', np.nan))

    # Check

    loc_missing_df = df[df['locseparate'].isna()]
    for i, row in loc_missing_df.iterrows():
        loc_len = row['loclen']
        if pd.notnull(loc_len):
            df.at[i, 'locseparate'] = 'Yes'
        else:
            df.at[i, 'locseparate'] = 'No'

    # If there are any "no"s for locseperate then the corresponding loclen values should be set to NaN as we give prority to the indicator
    df.loc[df['locseparate'] == 'No', 'loclen'] = np.nan

    if not suspected_loc:
        df.loc[df['locseparate'] == 'Suspected', 'loclen'] = np.nan

    return df

# Clean Mechanism of Injury Related Variables


def clean_mechanism_injury(df, extreme_caution=False):
    """
    Cleans mechanism of injury columns and imputes missing injury severity values
    based on mechanism types alone when detailed sub-variables are unavailable.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing mechanism and severity columns.
        extreme_caution (bool): If True, remaining unmapped missing values will be 
                                imputed as 'Severe'; if False, as 'Moderate'.

    Returns:
        pd.DataFrame: The cleaned DataFrame with imputed injury severity.
    """
    #  Map numeric codes to descriptive strings
    mech_map = {
        1.0: 'Occupant in MVC',
        2.0: 'Ped. Struck by Vehicle',
        3.0: 'Cyclist Struck by Vehicle',
        4.0: 'Bike Collison',
        5.0: 'Other Transportation Crash',
        6.0: 'Fall from Standing/walking',
        7.0: 'Running into Object',
        8.0: 'Fall from Elevation',
        9.0: 'Fall down Stairs',
        10.0: 'Sports',
        11.0: 'Assault',
        12.0: 'Struck by object',
        90.0: 'Other'
    }

    df['injurymech'] = df['injurymech'].replace(mech_map)
    df['high_impact_injsev'] = df['high_impact_injsev'].replace(
        {1.0: 'Low', 2.0: 'Moderate', 3.0: 'High'})

    #  Define High / Severe Severity Mechanisms
    high_mechanisms = [
        'Occupant in MVC',
        'Ped. Struck by Vehicle',
        'Cyclist Struck by Vehicle',
        'Fall from Elevation',
        'Fall down Stairs',
        'Struck by object'
    ]
    high_rule = df['injurymech'].isin(high_mechanisms)

    # Define Low Severity Mechanisms
    low_mechanisms = [
        'Fall from Standing/walking',
        'Running into Object'
    ]
    low_rule = df['injurymech'].isin(low_mechanisms)

    # Apply Imputations to Missing Severity Values
    missing_mask = df['high_impact_injsev'].isna()

    # Apply High Severity Imputations
    df.loc[missing_mask & high_rule, 'high_impact_injsev'] = 'High'

    # Apply Low Severity Imputations
    df.loc[missing_mask & low_rule, 'high_impact_injsev'] = 'Low'

    # Apply Fallback / Cautionary Imputations for Remaining Unmapped Cases
    still_missing = df['high_impact_injsev'].isna()
    default_severity = 'High' if extreme_caution else 'Moderate'
    df.loc[still_missing, 'high_impact_injsev'] = default_severity

    return df

# Clean Skull Fracture Related Variables (Inclduing Basilar fractures)


def clean_skull_fracture(df):
    """
    Cleans skull fracture related columns in the given DataFrame. First replaces numeric indicators with readable text strings.
    Then the function checks for inconsistencies between the indicator variable (sfxpalp and sfxbas) and the sub-variables (sfxpalpdepress, sfxbashem, sfxbasoto, sfxbasper, sfxbasret, sfxbasrhi). If there are any inconsistencies, then the function will impute the indicator variable based on the sub-variables. If any of the sub-variables are populated with a "Yes", then the indicator variable will be set to "Yes". If all of the sub-variables are populated with a "No", then the indicator variable will be set to "No". If all of the sub-variables are missing and indictaor is NaN, then for sfxpalp the indicator variable will be set to "Unclear".
    For sfxbas, if all of the sub-variables are missing and indictaor is NaN, then the indicator variable will be set to "No" as it is logical to assume that if there is no information about the sub-variables, then there is no skull fracture.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing skull fracture columns.
    Returns:
        pd.DataFrame: The cleaned DataFrame with skull fracture columns standardized.
    """
    skull_fracture_columns = ['sfxpalp', 'sfxpalpdepress', 'sfxbas',
                              'sfxbashem', 'sfxbasoto', 'sfxbasper', 'sfxbasret', 'sfxbasrhi']

    # Standardize numeric codes to Yes/No strings for all skull fracture columns
    df["sfxpalp"] = df["sfxpalp"].replace(
        {1.0: 'Yes', 0.0: 'No', 2.0: 'Unclear'})

    df["sfxpalpdepress"] = df["sfxpalpdepress"].replace(
        {1.0: 'Yes', 0.0: 'No', 92.0: np.nan})

    df["sfxbas"] = df["sfxbas"].replace({1.0: 'Yes', 0.0: 'No'})

    for col in ['sfxbashem', 'sfxbasoto', 'sfxbasper', 'sfxbasret', 'sfxbasrhi']:
        df[col] = df[col].replace({1.0: 'Yes', 0.0: 'No', 92.0: np.nan})

    for i, row in df.iterrows():
        # Check for inconsitencies in sfxpalp and its sub-variables
        if pd.isna(row['sfxpalp']):
            if row[['sfxpalpdepress']].eq('Yes').any():
                df.at[i, 'sfxpalp'] = 'Yes'
            else:
                df.at[i, 'sfxpalp'] = 'Unclear'
        # Make NaN no for sfxpalpdepress if sfxpalp is no or unclear since it is logical to assume that if there is no skull fracture, then there cannot be a depressed skull fracture.
        if row['sfxpalp'] in ['No', 'Unclear']:
            df.at[i, 'sfxpalpdepress'] = 'No'

        # Check for inconsistencies in sfxbas and its sub-variables
        if pd.isna(row['sfxbas']):
            if row[['sfxbashem', 'sfxbasoto', 'sfxbasper', 'sfxbasret', 'sfxbasrhi']].eq('Yes').any():
                df.at[i, 'sfxbas'] = 'Yes'
            elif row[['sfxbashem', 'sfxbasoto', 'sfxbasper', 'sfxbasret', 'sfxbasrhi']].eq('No').all():
                df.at[i, 'sfxbas'] = 'No'
            else:
                # Default to No if all sub-variables are missing
                df.at[i, 'sfxbas'] = 'No'

        # If sfxbas is No, then all sub-variables should be No as well since it is logical to assume that if there is no skull fracture, then there cannot be any of the sub-variables.
        if row['sfxbas'] == 'No':
            df.at[i, 'sfxbashem'] = 'No'
            df.at[i, 'sfxbasoto'] = 'No'
            df.at[i, 'sfxbasper'] = 'No'
            df.at[i, 'sfxbasret'] = 'No'
            df.at[i, 'sfxbasrhi'] = 'No'

    return df

# Clean Fontanelle Bulging Variable


def clean_fontbulg(df):
    """
    Cleans the fontanelle bulging variable ('fontbulg') by converting numeric flags to 'Yes'/'No' and imputes missing values based on age group.
    Takes Na values and maps to No due to rarity of fontanelle bulging across all age groups.


    Parameters:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame with imputed fontbulg values.
    """
    # Standardize numeric codes to Yes/No strings
    df['fontbulg'] = df['fontbulg'].replace({1.0: 'Yes', 0.0: 'No'})

    # Impute missing values
    df.loc[df['fontbulg'].isna(), 'fontbulg'] = 'No'

    return df

# Clean Acting Normal Varaible


def clean_actnorm(df):
    """
    Cleans the actnorm variable by converting numeric flags to 'Yes'/'No'   

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame with actnorm values standardized.
    """
    # Standardize numeric codes to Yes/No strings
    df['actnorm'] = df['actnorm'].replace({1.0: 'Yes', 0.0: 'No'})

    return df

# Clean Headache Related Variables


def clean_headache(df):
    """
    Cleans the headache related variables. Converts numeric flags to interpretible text strings.
    If headache is missing then check if severity is populated. If so then impute "Yes" to headache. 

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame with headache values standardized.
    """

    # Standardize numeric codes to Yes/No strings
    df['ha_verb'] = df['ha_verb'].replace(
        {1.0: 'Yes', 0.0: 'No', 91.0: "Non-verbal"})

    df['haseverity'] = df['haseverity'].replace(
        {1.0: 'Mild', 2.0: 'Moderate', 3.0: 'Severe', 92.0: np.nan})

    df['hastart'] = df['hastart'].replace(
        {1.0: 'Before Injury', 2.0: 'Within 1 Hour', 3.0: '1-4 Hours After', 4.0: 'More than 4 Hours After', 92.0: np.nan})

    # If headache is missing, check if severity is populated. If so, impute "Yes" to headache.
    df.loc[df['ha_verb'].isnull() & df['haseverity'].notnull(),
           'ha_verb'] = 'Yes'

    return df

# Clean vomiting related variables


def clean_vomit(df):
    """
    Cleans vomit related variables. First convert all numeric flags to interpretible text strings. 
    Then, for NaN values in 'vomit', check if any of the other vomit-related variables are populated. If so, impute "Yes" to 'vomit'. If all other vomit-related variables are NaN, then impute "No" to 'vomit'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame with vomit values standardized. 
    """
    # Convert numeric flags to text strings
    df['vomit'] = df['vomit'].replace({1.0: 'Yes', 0.0: 'No'})

    df['vomitnbr'] = df['vomitnbr'].replace(
        {1.0: 'Once', 2.0: 'Twice', 3.0: 'Three or more', 92.0: np.nan})

    df['vomitstart'] = df['vomitstart'].replace(
        {1.0: 'Before Injury', 2.0: 'Within 1 Hour', 3.0: '1-4 Hours After', 4.0: 'More than 4 Hours After', 92.0: np.nan})

    df['vomitlast'] = df['vomitlast'].replace(
        {1.0: 'Less than 1 Hour', 2.0: '1-4 Hours', 3.0: 'More than 4 Hours', 92.0: np.nan})

    # For NaN values in 'vomit', check if any of the other vomit-related variables are populated. If so, impute "Yes" to 'vomit'. If all other vomit-related variables are NaN, then impute "No" to 'vomit'.
    vomit_related_columns = ['vomitnbr', 'vomitstart', 'vomitlast']
    df.loc[df['vomit'].isnull(), 'vomit'] = df[vomit_related_columns].apply(
        lambda x: 'Yes' if x.notnull().any() else 'No', axis=1)

    return df

# Clean Dizzy Variable


def clean_dizzy(df):
    """
    Clean the dizzy variable, convert flags to text strings

    Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    df['dizzy'] = df['dizzy'].replace({1.0: 'Yes', 0.0: 'No'})

    return df

# Clean response


def clean_posintfinal(df):
    """
    Cleans the posintfinal variable by converting numeric flags to 'Yes'/'No'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame with posintfinal values standardized.
    """
    df['posintfinal'] = df['posintfinal'].replace({1.0: 'Yes', 0.0: 'No'})

    return df


def clean_data(df, include_severe_cases=False, suspected_loc=False, extreme_caution=False):
    """
    Cleans the entire DataFrame by applying all individual cleaning functions in sequence.
    In addition, we remove all obervatiosn for whihc the response (posintfinal) is missing.
    We then select only those varibales which are pertinent to the analysis. 

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        include_severe_cases (bool): Whether to include severe cases in GCS cleaning.
        suspected_loc (bool): Whether to treat suspected LOC as valid in LOC cleaning.
        extreme_caution (bool): Whether to impute remaining unmapped missing values as 'Severe' in mechanism of injury cleaning.

    Returns:
        pd.DataFrame: The fully cleaned DataFrame.

    """
    # Clean column names to ensure consistency
    df = df.clean_names()

    df = clean_age(df)
    df = clean_gcs(df, include_severe_cases)
    df = clean_ams(df)
    df = clean_hema(df)
    df = clean_loc(df, suspected_loc)
    df = clean_mechanism_injury(df, extreme_caution)
    df = clean_skull_fracture(df)
    df = clean_fontbulg(df)
    df = clean_actnorm(df)
    df = clean_headache(df)
    df = clean_vomit(df)
    df = clean_dizzy(df)
    df = clean_posintfinal(df)
    df = df[df['posintfinal'].notna()]

    df = df[[
        # Age
        'ageinmonth', 'ageinyears', 'agetwoplus',
        # GCS
        'gcstotal', 'gcseye', 'gcsverbal', 'gcsmotor', 'gcsgroup',
        # AMS
        'ams', 'amsagitated', 'amssleep', 'amsslow', 'amsrepeat', 'amsoth',
        # Hematoma
        'hema', 'hemaloc', 'hemasize',
        # LOC
        'locseparate', 'loclen',
        # Mechanism of Injury
        'injurymech', 'high_impact_injsev',
        # Skull Fracture
        'sfxpalp', 'sfxpalpdepress',
        # Basilar
        'sfxbas', 'sfxbashem', 'sfxbasoto', 'sfxbasper', 'sfxbasret', 'sfxbasrhi',
        # Fontanelle Bulging
        'fontbulg',
        # Acting Normal
        'actnorm',
        # Headache
        'ha_verb', 'haseverity', 'hastart',
        # Vomiting
        'vomit', 'vomitnbr', 'vomitstart', 'vomitlast',
        # Dizzy
        'dizzy',
        # Response
        'posintfinal'
    ]]

    return df
