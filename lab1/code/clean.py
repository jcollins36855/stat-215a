"""
clean.py

Whats in this file:

1   Creates COLUMN_RENAME_MAP. Column names were redone so that they
    Were more descriptive for ease of exploratory use.

2   A check was ran on columns that seemed to have a parent/child 
    relationship. For example, if it was reported that the patient
    did not have a seizure, then there should not be any markers 
    that indicate a description of a seizure (such as how long 
    the seizure lasted or when it occured). PARENT_CHILD_MAP was
    created to map these relationships then the function
    check_parent_child_logic checks to see if there are any places
    where a child is checked when a parent is not (indicating that
    the parent should be). It returns the columns and if there were 
    any violations in those columns.

3   A function called check_missing performs a thorough accounting of
    missing, NaN, and empty variables in each column. It creates a
    dataframe indicating if there were any flagged entries and 
    displays the column where the flagged entry exists.

4   The cleaning funciton gets to work checking that there are no duplicate
    rows and dropping them if they exist. Columns are renamed. Data
    exploration reveiled that there were float and integer types in the
    set. Since there is no need for a float value, they are all set to
    integers. Violations are checked and printed.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ---------------------------------------------------------------------------
# Reference only: what the special codes mean. Not used to recode anything —
# kept here so the meaning of 90/91/92 is documented alongside the pipeline
# that deliberately leaves them alone.
# ---------------------------------------------------------------------------
SPECIAL_CODES = {
    90: "Other",
    91: "Pre-verbal/Non-verbal",
    92: "Not applicable",
}


# ---------------------------------------------------------------------------
# Column renaming: raw PECARN variable names -> descriptive names.
# Columns not listed here keep their original raw name.
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "InjuryMech": "Injury_Mechanism",
    "Amnesia_verb": "Amnesia_verbal_patient",
    "LOCSeparate": "Loss_of_Consciousness_History",
    "LocLen": "Duration_of_Consciousness_Loss",
    "Seiz": "Post-Traumatic_Seizure",
    "SeizOccur": "When_Seizure_Occured",
    "SeizLen": "Length_of_Seizure",
    "ActNorm": "Acting_Normally",
    "HA_verb": "Headache_During_Eval",
    "HASeverity": "Headache_Severity",
    "HAStart": "Headache_Start_Time",
    "VomitStart": "Vomit_Start_Time",
    "VomitLast": "Last_Vomit_Ep_Time",
    "AMS": "Altered_Mental_Status(AMS)",
    "AMSAgitated": "AMS_Agitated",
    "AMSSleep": "AMS_Sleepy",
    "AMSSlow": "AMS_Slow_to_Respond",
    "AMSRepeat": "AMS_Repetative_Questions",
    "AMSOth": "AMS_Other",
    "SFxPalp": "Skull_Frax(SFx)_Palp",
    "SFxPalpDepress": "SFx_Palp_Depressed",
    "FontBulg": "Bulging_Anterior_Fontanelle",
    "SFxBas": "SFx_Basilar(Base)",
    "SFxBasHem": "SFx_Basilar_Hemotympanum(Eardrum)",
    "SFxBasOto": "SFx_Basilar_Oto(FluidfromEar)",
    "SFxBasPer": "SFx_Basilar_Per(EyeBruising)",
    "SFxBasRet": "SFx_Basilar_Ret(BehindEarBruising)",
    "SFxBasRhi": "SFx_Basilar_Rhi(FluidfromNose)",
    "Hema": "Hematoma",
    "HemaLoc": "Hematoma_Location",
    "HemaSize": "Hematoma_Size(SML)",
    "Clav": "Trauma_Above_Clavicles",
    "ClavFace": "Trauma_Face",
    "ClavNeck": "Trauma_Neck",
    "ClavFro": "Trauma_Frontal_Scalp",
    "ClavOcc": "Trauma_Occipital_Scalp",
    "ClavPar": "Trauma_Parietal_Scalp",
    "ClavTem": "Trauma_Temporal_Scalp",
    "NeuroD": "Neurological_Defficiency(ND)",
    "NeuroDMotor": "NeuroD_Motor",
    "NeuroDSensory": "NeuroD_Sensory",
    "NeuroDCranial": "NeuroD_Cranial(+pupilreactivity)",
    "NeuroDReflex": "NeuroD_Reflex",
    "NeuroDOth": "NeuroDOther",
    "OSI": "Other_Substantial_Injury(OSI)",
    "OSIExtremity": "OSI_Extremity",
    "OSICut": "OSI_Laceration",
    "OSICspine": "OSI_C-Spine",
    "OSIFlank": "OSI_Chest/Back/Flank",
    "OSIAbdomen": "OSI_Intra-Abdomen",
    "OSIPelvis": "OSI_Pelvis",
    "OSIOth": "OSI_Other",
    "Drugs": "Suspicion_of_Drugs",
    "CTForm1": "CT_XRay_or_MRI_Ordered?",
    "IndAge": "Age_Indicated_in_CTDecision",
    "IndAmnesia": "Amnesia_Indicated_in_CTDecision",
    "IndAMS": "AMS_Indicated_in_CTDecision",
    "IndClinSFx": "SFx_Indicated_in_CTDecision",
    "IndHA": "Headache_Inidcated_in_CTDecision",
    "IndHema": "Hematoma_Indicated_in_CTDecision",
    "IndLOC": "Loss_of_Consciousness_Indicated_in_CTDecision",
    "IndMech": "Injury_Mechanism_Indicated_in_CTDecision",
    "IndNeuroD": "NeuroD_Indicated_in_CTDecision",
    "IndRqstMD": "MDRqst_Indicatedd_In_CTDecision",
    "IndRqstParent": "Parental_AnxietyorRequest_Indicated_in_CTDecision",
    "IndRqstTrauma": "Trauma_Team_Requested_CTDecsion",
    "IndSeiz": "Seizure_Indicated_in_CTDecision",
    "IndVomit": "Vomit_Indicated_in_CTDecision",
    "IndXraySFx": "SFxXray_Indicated_in_CTDecision",
    "IndOth": "Other_Indicated_in_CTDecision",
    "CTSed": "Sedation_Used_forCT",
    "CTSedAgitate": "Sedation_bc_AgitationCT",
    "CTSedAge": "Sedation_bc_AgeCT",
    "CTSedRqst": "Sedation_bc_TechnicianRequestCT",
    "Observed": "Observed_in_ED_Determined_CTDecision",
    "EDDisposition": "Destination_After_ED",
    "CTDone": "CT_Performed",
    "EDCT": "CT_Done_in_ED",
    "PosCT": "Traumatic_Brain_Injury_on_CT",
    "Finding1": "F1_Cerebellar_Hemorrhage(lowerbrainbleed)",
    "Finding2": "F2_Cerebral_Contusion(bruise)",
    "Finding3": "F3_Cerebral_Edema(brainswelling)",
    "Finding4": "F4_Intracerebral_Hematoma(bloodclotinbrain)",
    "Finding5": "F5_Skull_Diastasis(abnormalwidening)",
    "Finding6": "F6_Epidural_Hematoma(bloodbetweenduraandskull)",
    "Finding7": "F7_Extra-Axial_Hematoma(bloodbetweenbrainandskull)",
    "Finding8": "F8_Intraventricular_Hemorrhage(bloodwithinbrain)",
    "Finding9": "F9_Brain_Shift(offcenter)",
    "Finding10": "F10_Pneumocephalus(airinbrain)",
    "Finding11": "F11_Skull_Fraxure",
    "Finding12": "F12_Subarachnoid_Hemorrhage(bloodbetweenbrainanddura)",
    "Finding13": "F13_Subdural_Hematoma(bloodbetweenarachnoidanddura)",
    "Finding14": "F14_Infarction(lackofblood)",
    "Finding20": "F15_Diffuse_Axonal_Injury(brokennerves)",
    "Finding21": "F16_Herniation(buldingbraintissue)",
    "Finding22": "F17_Shear_Injury(sameasdiffuseaxonal)",
    "Finding23": "F18_Sigmoid_Sinus_Thrombosis(bloodclotinsinus)",
    "DeathTBI": "Death_From_TBI",
    "HospHead": "Hospital_2ormore_Nights",
    "HospHeadPosCT": "Hospital_2ormore_Nights_TBIonCT",
    "Intub24Head": "Intubated_24hrormore",
    "PosIntFinal": "Clinically-Important_TBI",
}


# ---------------------------------------------------------------------------
# Parent -> child column groups, for skip-logic validation.
# Column names here refer to POST-RENAME names (see COLUMN_RENAME_MAP above).
# ---------------------------------------------------------------------------
PARENT_CHILD_MAP = {
    "Loss_of_Consciousness_History": ["Duration_of_Consciousness_Loss"],
    "Post-Traumatic_Seizure": ["When_Seizure_Occured", "Length_of_Seizure"],
    "Headache_During_Eval": ["Headache_Severity", "Headache_Start_Time"],
    "Vomit": ["VomitNbr", "Vomit_Start_Time", "Last_Vomit_Ep_Time"],
    "Altered_Mental_Status(AMS)": [
        "AMS_Agitated", "AMS_Sleepy", "AMS_Slow_to_Respond",
        "AMS_Repetative_Questions", "AMS_Other",
    ],
    "Skull_Frax(SFx)_Palp": ["SFx_Palp_Depressed"],
    "SFx_Basilar(Base)": [
        "SFx_Basilar_Hemotympanum(Eardrum)",
        "SFx_Basilar_Oto(FluidfromEar)",
        "SFx_Basilar_Per(EyeBruising)",
        "SFx_Basilar_Ret(BehindEarBruising)",
        "SFx_Basilar_Rhi(FluidfromNose)",
    ],
    "Hematoma": ["Hematoma_Location", "Hematoma_Size(SML)"],
    "Trauma_Above_Clavicles": [
        "Trauma_Face", "Trauma_Neck", "Trauma_Frontal_Scalp",
        "Trauma_Occipital_Scalp", "Trauma_Parietal_Scalp", "Trauma_Temporal_Scalp",
    ],
    "Neurological_Defficiency(ND)": [
        "NeuroD_Motor", "NeuroD_Sensory", "NeuroD_Cranial(+pupilreactivity)",
        "NeuroD_Reflex", "NeuroDOther",
    ],
    "Other_Substantial_Injury(OSI)": [
        "OSI_Extremity", "OSI_Laceration", "OSI_C-Spine",
        "OSI_Chest/Back/Flank", "OSI_Intra-Abdomen", "OSI_Pelvis", "OSI_Other",
    ],
    "CT_XRay_or_MRI_Ordered?": [
        "Age_Indicated_in_CTDecision", "Amnesia_Indicated_in_CTDecision",
        "AMS_Indicated_in_CTDecision", "SFx_Indicated_in_CTDecision",
        "Headache_Inidcated_in_CTDecision", "Hematoma_Indicated_in_CTDecision",
        "Loss_of_Consciousness_Indicated_in_CTDecision",
        "Injury_Mechanism_Indicated_in_CTDecision", "NeuroD_Indicated_in_CTDecision",
        "MDRqst_Indicatedd_In_CTDecision",
        "Parental_AnxietyorRequest_Indicated_in_CTDecision",
        "Trauma_Team_Requested_CTDecsion", "Seizure_Indicated_in_CTDecision",
        "Vomit_Indicated_in_CTDecision", "SFxXray_Indicated_in_CTDecision",
        "Other_Indicated_in_CTDecision",
    ],
    "Sedation_Used_forCT": [
        "Sedation_bc_AgitationCT", "Sedation_bc_AgeCT",
        "Sedation_bc_TechnicianRequestCT",
    ],
    "CT_Performed": [
        "CT_Done_in_ED", "Traumatic_Brain_Injury_on_CT",
        "F1_Cerebellar_Hemorrhage(lowerbrainbleed)", "F2_Cerebral_Contusion(bruise)",
        "F3_Cerebral_Edema(brainswelling)", "F4_Intracerebral_Hematoma(bloodclotinbrain)",
        "F5_Skull_Diastasis(abnormalwidening)", "F6_Epidural_Hematoma(bloodbetweenduraandskull)",
        "F7_Extra-Axial_Hematoma(bloodbetweenbrainandskull)",
        "F8_Intraventricular_Hemorrhage(bloodwithinbrain)", "F9_Brain_Shift(offcenter)",
        "F10_Pneumocephalus(airinbrain)", "F11_Skull_Fraxure",
        "F12_Subarachnoid_Hemorrhage(bloodbetweenbrainanddura)",
        "F13_Subdural_Hematoma(bloodbetweenarachnoidanddura)",
        "F14_Infarction(lackofblood)", "F15_Diffuse_Axonal_Injury(brokennerves)",
        "F16_Herniation(buldingbraintissue)", "F17_Shear_Injury(sameasdiffuseaxonal)",
        "F18_Sigmoid_Sinus_Thrombosis(bloodclotinsinus)",
    ],
}


# ---------------------------------------------------------------------------
# Employee and Certification Type Mapping
# ---------------------------------------------------------------------------
empl_type_mapping = {
    1: "Nurse Practitioner",
    2: "Physician Assistant",
    3: "Resident",
    4: "Fellow",
    5: "Faculty",
}

certification_mapping = {
    1: "Emergency Medicine",
    2: "Pediatrics",
    3: "Pediatrics Emergency Medicine",
    4: "Emergency Medicine and Pediatrics",
    90: "Other",
}


##############################################################################

# Rename columns
def rename_columns(df, rename_map=COLUMN_RENAME_MAP):
    """
    Rename columns using the provided mapping. Columns not in rename_map
    keep their original name.
    """
    return df.rename(columns=rename_map)


########################################3

#Filtering low GCS scores
def filter_low_severity(df, gcs_col="GCSTotal", min_gcs=14, verbose=True):
    """
    Restrict the dataframe to lower-severity patients (GCS >= min_gcs),
    matching Kuppermann et al.'s inclusion criteria (GCS 14-15).
+
    """

    filtered = df[df[gcs_col] >= min_gcs].copy()
 
    return filtered

#########################################################################################
# Originally checked for 99 and 9999, but those would be reasonable values in some columns
# Such as PatNum. A manual check of other columns did not show the existence of these 
# salues elsewhere.
def check_missing(raw_df, placeholder_values=("NA", "N/A", "na", "n/a", ".", "-", "?")):
    """
    What it does:
    Checks for missing values, NaN, whitespace-only,
    strings, or common placeholder strings that look like real values.

    Returns:
    A DataFrame with one row per column with any flagged values

    """
    nan_counts = raw_df.isna().sum()

    def _count_blank_strings(col):
        if col.dtype == object:
            return col.astype(str).str.strip().eq("").sum()
        return 0

    def _count_placeholders(col):
        return col.astype(str).isin(placeholder_values).sum()

    blank_string_counts = raw_df.apply(_count_blank_strings)
    placeholder_counts = raw_df.apply(_count_placeholders)

    summary = pd.DataFrame({
        "NaN Value": nan_counts,
        "Blank String": blank_string_counts,
        "Placeholder String": placeholder_counts,
    })
    summary["Total Flagged"] = summary.sum(axis=1)
    summary = summary.sort_values("Total Flagged", ascending=False)

    return summary[summary["Total Flagged"] > 0]

##########################################################################################

# Employee type chart
def empl_type_table(pud, empl_type_mapping):
    pud_mapped = pud.copy()
    pud_mapped['EmplType'] = pud_mapped['EmplType'].map(empl_type_mapping)

    return pud_mapped.groupby('EmplType', dropna=False).size().reset_index(name='Count')


###############
# Employee Missingness Assessment Graph:

def employee_missingness_plot(pud_precleaned, empl_type_mapping, certification_mapping):
    import os
    os.makedirs("../figs", exist_ok=True)

    # Map Values
    pud_mapped = pud_precleaned.copy()
    pud_mapped['EmplType'] = pud_mapped['EmplType'].map(empl_type_mapping)
    pud_mapped['Certification'] = pud_mapped['Certification'].map(certification_mapping)

    pud_clean = pud_mapped.dropna(subset=['EmplType', 'Certification']).copy()

    # Determine the total number of columns being checked for NaNs
    features_to_check = [col for col in pud_clean.columns if col not in ['EmplType', 'Certification']]
    num_features = len(features_to_check)

    # Calculate NaN count and convert to a percentage
    pud_clean['NaN_Count'] = pud_clean[features_to_check].isna().sum(axis=1)
    pud_clean['NaN_Pct'] = (pud_clean['NaN_Count'] / num_features) * 100

    # Grab missingness by Employee Type 
    summary_stats = pud_clean.groupby('EmplType')['NaN_Pct'].agg(['mean', 'median', 'count']).sort_values(by='mean', ascending=False)

    # Typography (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 3. Create a clean bar chart visualization using pure Matplotlib
    plt.figure(figsize=(6, 5))

    bars = plt.bar(summary_stats.index, summary_stats['mean'], 
        color='#1B365D', 
        edgecolor='black',
        width=0.6
    )

    plt.title('Average Percentage of Missing Values per Record by Employee Type', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Employee Type', fontsize=14, labelpad=10)
    plt.ylabel('Mean Percentage of \nMissing Values per Row (%)', fontsize=14, labelpad=10)
    plt.xticks(rotation=20, ha='right', fontsize=12)
    plt.ylim(0, 1.6)
    plt.yticks(fontsize=12)
    plt.tick_params(axis="both", direction="in") 

    # Add grid behind the bars for readability
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    #  Put behind data
    plt.gca().set_axisbelow(True)

    # Inward ticks (from lab slides)
    plt.tick_params(
        axis='both', 
        direction='in', 
        top=True, 
        right=True, 
        which='both'
    )

    # Add numbers at top of bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig("../figs/employee_type_missingness.pdf", dpi=200, bbox_inches="tight", format="pdf")
    plt.show()


#####################################################################################

# This is a funciton that looks at parent and child column logic.

# If a parent is negative, or has a no (0),
# Then all of its children should be not applicable (92) as well. 
# Otherwise, if a child column has a yes (1), then the parent column
# should have a yes as well(1)

# If a parent is positive, or has a yes (1),
# then the child columns should either be no (0) or yes (1).
# Some child columns may have other positive numbers such as 
# 2, 3, 4, and 5 if the parent is positive. 

# Effectively a 92, meaning 'not applicable' on a positive
# parent column should be a 0.
# We will also make the assumption that a NaN value 
# should be a 0. 

# If there is a 92 or a NaN value where a parent column is yes (1), 
# it should be marked as a no (0). 
# We assume that if the column is skipped (NaN), it is likley becuase 
# the patient does not present with that symptom.

# *Exception: Loss_of_Conscious_History, which can have a 2 and
# if it does have a 2, then its child could be 92 or another number.

def check_parent_child_logic(df, parent_child_map=PARENT_CHILD_MAP, verbose=True):
  """Audits parent-child skip-logic consistency without modifying the dataframe.
    
    What it does:
  Checks for three main rule violations: 
  1. Parent is 0 ("No"), but any child is not 92 ("Not applicable"). 
  2. Any child is 1 ("Yes"), but the parent is not 1.
  3. Parent is 1 ("Yes"), but any child has a 92 or NaN (should be 0 or active
    values).

    Returns:
  Includes explicit exception handling for Loss_of_Conscious_History (allowing
  parent = 2).
  """
  violations = {}

  for parent_col, child_cols in parent_child_map.items():
    missing_cols = [c for c in [parent_col] + child_cols if c not in df.columns]
    if missing_cols:
      if verbose:
        print(f"[{parent_col}] skipped — column(s) not found: {missing_cols}")
      continue

    valid_children = [c for c in child_cols if c in df.columns]
    if not valid_children:
      continue

    # Allow parent = 2 as a valid active parent specifically for Loss of Consciousness
    valid_parents = (
        [1, 2] if parent_col == "Loss_of_Consciousness_History" else [1]
    )

    # Rule 1: Parent is 0, but any child is not 92
    v_parent_zero = (df[parent_col] == 0) & df[valid_children].ne(92).any(axis=1)

    # Rule 2: Any child is 1, but parent is not a valid active parent
    v_child_one = df[valid_children].eq(1).any(axis=1) & ~df[parent_col].isin(
        valid_parents
    )

    # Rule 3: Parent is a valid active parent, but any child has 92 or NaN
    v_parent_one = df[parent_col].isin(valid_parents) & (
        df[valid_children].eq(92) | df[valid_children].isna()
    ).any(axis=1)

    # Combine violations for this group (duplicates dropped if a row triggers multiple rules)
    group_violations = df[
        v_parent_zero | v_child_one | v_parent_one
    ].drop_duplicates()
    violations[parent_col] = group_violations

    if verbose:
      print(f"[{parent_col}] violations found: {len(group_violations)}")

  return violations

###############################################################################################

def impute_parent_child_empirical(
    df,
    parent_col,
    child_cols,
    active_parent_vals=[1],
    skip_parent_val=0,
    skip_val=92,
):
  """
    What it does:
    This code looks at:
    1. Where the data has NaN values in the child columns of present parents (a violation) and
    2. Where the data is fully filled out and clean

    It then takes the clean data and computes the proportion of child values for each 
    unique parent value.
    Then it applies these proportions to the values of the data where there are NaN
    values. 

    For parents that have more than one child, the joint distribution proportions are
    calculated. So if parent column A has children B and C, where B and C should be 0 or 1,
    The function looks at the clean data and caluclates the proportion of values where 
    there is a [0, 1], [0, 0], [1, 0], or [1, 1] entry and uses that to fill in NaN values
    in the other part of the dataset. This is important becuase some of the child columns
    are dependent on each other.

    It returns:
    A clean data frame where parent and child 
  """
  df = df.copy()
  if isinstance(child_cols, str):
    child_cols = [child_cols]

  for p_val in active_parent_vals:
    missing_mask = (df[parent_col] == p_val) & df[child_cols].isna().any(axis=1)
    n_missing = missing_mask.sum()
    if n_missing == 0:
      continue

    # Isolate valid rows where parent == p_val (active parent values) and ALL children are present
    valid_mask = (df[parent_col] == p_val) & df[child_cols].notna().all(axis=1)
    valid_data = df[valid_mask]

    if len(valid_data) == 0:
      continue

    #Extract joint distribution with proportions
    joint_dist = (
        valid_data[child_cols]
        .value_counts(normalize=True)
        .reset_index(name="proportion")
    )

    if len(joint_dist) == 0:
      continue

    # Sample complete rows cleanly using pandas native sampling
    sampled_subset = joint_dist.sample(
        n=n_missing, weights="proportion", replace=True
    )

    # Align the sampled rows to match the exact index of the missing rows in df
    sampled_subset.index = df[missing_mask].index

    # Assign
    for col in child_cols:
      df.loc[missing_mask, col] = sampled_subset[col]

  # If the parent indicates non-occurrence (0) automatically assign the designated 
  # skip code (e.g., 92) to any missing or unrecorded child columns
  if skip_parent_val is not None:
    skip_mask = (df[parent_col] == skip_parent_val) & df[child_cols].isna().any(
        axis=1
    )
    if skip_mask.sum() > 0:
      df.loc[skip_mask, child_cols] = skip_val

  return df

#####################################################################################

def plot_skull_fracture_citbi_by_age(pud, sfx_col="Skull_Frax(SFx)_Palp",
                                       citbi_col="Clinically-Important_TBI",
                                       age_col="AgeTwoPlus",
                                       fracture_label="Palpable Skull Fracture",
                                       save_path="../figs/finding1_skull_fracture_citbi_by_age.png"):
    """
    Plot the rate of clinically-important TBI (ciTBI), split by age group
    (<2 years vs >=2 years) and by palpable skull fracture status
    (1 or 0 only, must have valid code).

    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # To specify, we are only looking at when there is a clear
    # occurance or not of a palpable skull fracture, and a valid age code
    valid = pud[
        pud[sfx_col].isin([0, 1]) &
        pud[citbi_col].isin([0, 1]) &
        pud[age_col].isin([1, 2])
    ]

    # Compute ciTBI rate for each age group / skull fracture combination
    grouped = valid.groupby([age_col, sfx_col])[citbi_col].agg(['mean', 'count'])
    grouped['mean'] = grouped['mean'] * 100

    # AgeTwoPlus is coded 1 = <2 years, 2 = >=2 years, per the study codebook
    order = [(1, 0), (1, 1), (2, 0), (2, 1)]
    labels = ["$<$2 yrs\nNo Fracture", "$<$2 yrs\nFracture",
          "$\\geq$2 yrs\nNo Fracture", "$\\geq$2 yrs\nFracture"]
    colors = ["#1f4e79", "#c0392b", "#1f4e79", "#c0392b"]

    rates = [grouped.loc[key, 'mean'] for key in order]
    counts = [grouped.loc[key, 'count'] for key in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, rates, color=colors, edgecolor="black", width=0.6)

    # Annotate each bar with rate and sample size
    for bar, rate, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{rate:.1f}%\n(n={n:,})", ha="center", fontsize=14)

    ax.set_xlabel(f"Age Group and {fracture_label} Status", fontsize=16)
    ax.set_ylabel("Rate of Clinically-Important \nTBI (%)", fontsize=16)
    ax.set_ylim(0, max(rates) * 1.05)  #Ensure no title and bar label interfere
    ax.set_title(f"Percent of Patients in Trial with CiTBI vs Evidence of {fracture_label} \nAcross Both Age Groups", 
                 fontsize=18,
                 pad = 25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=12, top=False, right=False)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    #  Put behind data
    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(save_path, dpi=200, format="pdf")
    plt.show()

    return grouped

#######################################################################################

def plot_predictor_correlation_matrix(pud, cols=None,
                                        save_path="../figs/finding2_predictor_correlation.png"):
    """
    Plot a correlation matrix (heatmap!) among key clinical predictors from Kupperman,
    other parent columns that they might have missed, and
    the ciTBI outcome. Only rows with valid 0/1 codes across all selected
    columns are used.

    Parent columns added:
     - Post-Traumatic_Seizure
     - Trauma_Above_Clavicles
     - Neurological_Defficiency(ND)
     - Other_Substantial_Injury(OSI)

    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    if cols is None:
        # Core predictors from Kuppermann's derived trees, plus outcome
        cols = [
            "Skull_Frax(SFx)_Palp",
            "Altered_Mental_Status(AMS)",
            "Loss_of_Consciousness_History",
            "Vomit",
            "Headache_During_Eval",
            "Acting_Normally",
            "Clinically-Important_TBI",
            "Post-Traumatic_Seizure",
            "Trauma_Above_Clavicles",
            "Neurological_Defficiency(ND)",
            "Other_Substantial_Injury(OSI)"
        ]

    # Keep only rows where every selected column has a valid 0/1 response
    valid = pud[pud[cols].isin([0, 1]).all(axis=1)][cols]

    corr = valid.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.tick_params(axis="both", direction="in") 
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(cols, fontsize=10)

    # Annotate each cell with its correlation value
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.values[i, j]) > 0.5 else "black",
                     fontsize=9)

    ax.set_title("Correlation Among Key Clinical Predictors and ciTBI", fontsize=15, pad=15)
    fig.colorbar(im, ax=ax, label="Pearson Correlation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, format="pdf")
    plt.show()

    return corr

#####################################################################################

def count_by_gcs_threshold(df, gcs_col="GCSTotal"):
    """
    Count how many rows have GCS >= 14 vs GCS < 14.

    """
    above = (df[gcs_col] >= 14).sum()
    below = (df[gcs_col] < 14).sum()
    print(f"GCS >= {14}: {above:,}")
    print(f"GCS < {14}: {below:,}")
    return {"GCS Scores >= 14 (less severe)": above, "GCS Scores < 14 (more severe)": below}


#######################################################################3

def generate_modeling_variants(df, parent_child_map=PARENT_CHILD_MAP):
    """
    Takes the audited dataframe and outputs two distinct cleaned versions:
    1. df_tree: Preserves 92s as a distinct category (ideal for Random Forest / XGBoost).
    2. df_linear: Maps 92s and NaNs to 0 where parents are 0 (ideal for Logistic Regression).
    
    Returns
    two different dataframes (df_tree, df_linear)
    """
    df = df.copy()
    
    # Create the linear regression version (recodes 92s/NaNs to 0 based on skip-logic)
    df_linear = df.copy()
    
    # Create the tree-based version (keeps 92s intact, only fixing hard rule inversions like child=1 forcing parent=1)
    df_tree = df.copy()
    
    for parent_col, child_cols in parent_child_map.items():
        if parent_col not in df_linear.columns:
            continue
        valid_children = [c for c in child_cols if c in df_linear.columns]
        if not valid_children:
            continue
            
        # If any child is 1, the parent must be 1
        child_has_one = df_linear[valid_children].eq(1).any(axis=1)
        fix_parent_mask = child_has_one & (df_linear[parent_col].eq(0) | df_linear[parent_col].isna())
        
        df_linear.loc[fix_parent_mask, parent_col] = 1
        df_tree.loc[fix_parent_mask, parent_col] = 1

        # If a Parent is 0, Children must be 92
        parent_is_zero = df_linear[parent_col].eq(0)
        for c in valid_children:
            df_tree.loc[parent_is_zero, c] = 92
            df_linear.loc[parent_is_zero, c] = 92  # Temporarily set to 92, adjusted for linear next

        # Linear-specific adjustment (Recode 92 to 0 for regression math)
        # For linear models, a structural skip (92) when parent is 0 should be 0, 
        # while preserving the distinction that the parent was 0.
        parent_is_negative_or_zero = df_linear[parent_col].eq(0) | df_linear[parent_col].isna()
        for c in valid_children:
            # Recode structural 92s to 0 for linear model compatibility
            df_linear.loc[parent_is_negative_or_zero & df_linear[c].eq(92), c] = 0
            # Handle NanS
            df_linear[c] = df_linear[c].fillna(0)

    return df_tree, df_linear

#######################################################################################

def fit_logistic_regression(df_linear, target="Clinically-Important_TBI",
                              predictors=None, verbose=True,
                              save_path="../figs/modeling_logistic_coefficients.pdf"):
    """
    Fit a logistic regression predicting ciTBIs,
    using the linear-model-ready dataframe (92s recoded to 0).

    """
    import os
    from sklearn.linear_model import LogisticRegression

    if predictors is None:
        predictors = [
            "Skull_Frax(SFx)_Palp", "Altered_Mental_Status(AMS)",
            "Loss_of_Consciousness_History", "Vomit",
            "Headache_During_Eval", "Acting_Normally",
        ]

    valid = df_linear[
        df_linear[target].isin([0, 1]) &
        df_linear[predictors].notna().all(axis=1) # Drops NaN parent rows
    ]
    X = valid[predictors]
    y = valid[target]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    if verbose:
        print("Logistic Regression coefficients:")
        for name, coef in zip(predictors, model.coef_[0]):
            print(f"  {name}: {coef:.3f}")

    # Plot coefficients as a horizontal bar chart, sort by magnitude
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    coefs = model.coef_[0]
    order = sorted(range(len(coefs)), key=lambda i: coefs[i])
    sorted_names = [predictors[i] for i in order]
    sorted_coefs = [coefs[i] for i in order]
    colors = ["#1f4e79" if c < 0 else "#c0392b" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(sorted_names, sorted_coefs, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("Predictors of Clinically-Important TBI")
    ax.tick_params(axis="both", direction="in")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, format="pdf")
    plt.show()

    return model

################################

def fit_decision_tree(df_tree, target="Clinically-Important_TBI",
                        predictors=None, max_depth=3,
                        save_path="../figs/modeling_decision_tree.pdf"):
    """
    Fit a shallow decision tree predicting ciTBI,
    using the tree-ready dataframe (92s preserved as their own category).
    Plots and saves the tree structure.

    """
    import os
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if predictors is None:
        predictors = [
            "Skull_Frax(SFx)_Palp", "Altered_Mental_Status(AMS)",
            "Loss_of_Consciousness_History", "Vomit",
            "Headache_During_Eval", "Acting_Normally",
        ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    valid = df_tree[
        df_tree[target].isin([0, 1]) &
        df_tree[predictors].notna().all(axis=1)
    ]
    X = valid[predictors]
    y = valid[target]

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    model.fit(X, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    artists = plot_tree(model, feature_names=predictors, class_names=["No ciTBI", "ciTBI"],
                         filled=True, fontsize=9, ax=ax)

    # Recolor each box by its actual ciTBI rate, using a clearer colormap
    cmap = plt.get_cmap("YlOrRd")
    node_values = model.tree_.value  # shape (n_nodes, 1, n_classes)
    rates = [v[0][1] / v[0].sum() for v in node_values]
    max_rate = max(rates) if max(rates) > 0 else 1

    for artist, rate in zip(artists, rates):
        bbox = artist.get_bbox_patch()
        if bbox is not None:
            bbox.set_facecolor(cmap(rate / max_rate))

    # Add a colorbar legend showing what the colors mean
    norm = mpl.colors.Normalize(vmin=0, vmax=max_rate * 100)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("ciTBI Rate (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(f"Decision Tree for ciTBI Prediction (max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, format="pdf")
    plt.show()

    return model


##################################################################################3

def clean_data(df, verbose=True):
    """
    Clean the pud dataframe. Runs check_missing on the raw data, then
    drops duplicates, renames columns, casts to nullable Int64, and runs
    check_parent_child_logic on the renamed data.
    """
    df = df.copy()

    # 1. Check for disguised missingness on the RAW data, before anything
    #    else touches it (blank/placeholder strings are only reliably
    #    detectable while columns are still their original dtype).
    if verbose:
        print("Missingness check on raw data:")
        missing_report = check_missing(df)
        if missing_report.empty:
            print("  None found.")
        else:
            print(missing_report)
    else:
        check_missing(df)

    # 2. Drop duplicate rows, if any.
    n_before = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"\nDropped {n_before - len(df)} duplicate row(s).")

    # 3. Rename columns to descriptive names.
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # 4. Cast to nullable integer dtype (keeps real NaNs alongside ints).
    df = df.astype("Int64")

    # 5. Validate parent/child skip-logic on the renamed data — report only.
    if verbose:
        print("\nParent/child skip-logic check:")
    violations = check_parent_child_logic(df, verbose=verbose)
    n_violations = sum(len(v) for v in violations.values())
    if n_violations and verbose:
        print(f"\nWARNING: {n_violations} total skip-logic violations found across all groups.")

    return df

