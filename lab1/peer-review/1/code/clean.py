#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(raw, codebook):
    clean = raw.copy()

    na92_vars = codebook[
        codebook["Values"].astype(str).str.contains(
            r"92\s+Not applicable",
            case=False,
            na=False,
            regex=True
        )
    ]["Variable"].tolist()

    for col in na92_vars:
        if col in clean.columns:
            clean[col] = clean[col].replace(92, np.nan)

    return clean

# In[4]:


data_dir = Path(__file__).resolve().parent.parent / "data"
raw = pd.read_csv(data_dir / "TBI PUD 10-08-2013.csv")

# In[5]:

doc = pd.read_excel(data_dir / "TBI PUD Documentation 10-08-2013.xlsx")


# In[6]:


raw.shape


# In[7]:


raw.head()


# In[8]:


raw.dtypes


# In[9]:


raw.columns.tolist()


# In[10]:


doc.shape


# In[11]:


doc.head(20)


# In[12]:


doc.columns.tolist()


# In[13]:


codebook = doc.iloc[10:, :4].copy()

codebook.columns = [
    "Variable",
    "Description",
    "Values",
    "Notes"
]

codebook.head(10)


# In[14]:


raw_vars = set(raw.columns)
doc_vars = set(codebook["Variable"].dropna())
raw_vars - doc_vars


# In[15]:


doc_vars - raw_vars


# In[16]:


raw.duplicated().sum()


# In[17]:


raw["PatNum"].duplicated().sum()


# In[18]:


missing = raw.isna().mean().sort_values(ascending=False)

missing.head(20)


# ### Initial data quality checks
# 
# The raw dataset contains 43,399 observations and 125 variables. All variables in the raw data are documented in the codebook, and there are no duplicate rows or duplicate patient IDs. Initial missingness checks show substantial variation across variables, but these rates only reflect values represented as `NaN`; special numeric codes must also be interpreted using the documentation.

# In[19]:


raw["Amnesia_verb"].value_counts(dropna=False)


# In[20]:


raw["LOCSeparate"].value_counts(dropna=False)


# In[21]:


raw["LocLen"].value_counts(dropna=False)


# In[22]:


raw["Seiz"].value_counts(dropna=False)


# In[23]:


pd.crosstab(
    raw["LOCSeparate"],
    raw["LocLen"],
    dropna=False
)


# In[24]:


pd.crosstab(
    raw["Seiz"],
    raw["SeizOccur"],
    dropna=False
)


# LOCSeparate/LocLen and Seiz/SeizOccur were internally consistent: observations without the condition used the not-applicable code for the follow-up variable. However, the follow-up variable was sometimes missing even when the condition was present.

# In[25]:


for col in [
    "AgeinYears",
    "Gender",
    "Race",
    "Ethnicity",
    "GCSTotal",
    "CTDone",
    "PosCT",
    "Finding1"
]:
    print("\n", col)
    print(raw[col].value_counts(dropna=False).sort_index())


# In[26]:


raw["AgeinYears"].describe()


# In[27]:


codebook[
    codebook["Variable"].isin(
        ["Race", "Ethnicity", "CTDone", "PosCT", "Finding1"]
    )
]


# In[28]:


pd.crosstab(
    raw["CTDone"],
    raw["PosCT"],
    dropna=False
)


# In[29]:


pd.crosstab(
    raw["CTDone"],
    raw["Finding1"],
    dropna=False
)


# logical consistency check: no issues so far

# In[30]:


special_codes = codebook[
    codebook["Values"].astype(str).str.contains(r"\b90\b|\b91\b|\b92\b", regex=True)
]

special_codes[["Variable", "Description", "Values", "Notes"]]


# In[31]:


special_in_data = []

for col in raw.columns:
    if pd.api.types.is_numeric_dtype(raw[col]):
        vals = set(raw[col].dropna().unique())
        if any(v in vals for v in [90, 91, 92]):
            special_in_data.append(col)

special_in_data


# In[32]:


special_doc = codebook[
    codebook["Values"].astype(str).str.contains(
        "Not applicable|Unknown|Pre-verbal|Other",
        case=False,
        regex=True
    )
]

special_doc[["Variable", "Description", "Values", "Notes"]]


# In[33]:


not_applicable = codebook[
    codebook["Values"].astype(str).str.contains(
        "Not applicable", case=False, na=False
    )
]

not_applicable[["Variable", "Values"]]


# In[34]:


unknown_codes = codebook[
    codebook["Values"].astype(str).str.contains(
        "Unknown", case=False, na=False
    )
]

unknown_codes[["Variable", "Values"]]


# In[35]:


clean = clean_data(raw, codebook)


# In[38]:


clean["PosCT"].value_counts(dropna=False)


# In[39]:


clean_missing = clean.isna().mean().sort_values(ascending=False)
clean_missing.head(20)


# In[40]:


comparison = pd.DataFrame({
    "raw_missing": raw.isna().mean(),
    "clean_missing": clean.isna().mean()
})

comparison["increase"] = (
    comparison["clean_missing"] - comparison["raw_missing"]
)

comparison.sort_values("increase", ascending=False).head(20)


# In[41]:


raw_missing = raw.isna().mean().sort_values(ascending=False)
raw_missing.head(15)


# In[42]:


clean.shape


# In[43]:


clean[["AgeinYears", "GCSTotal", "Gender", "Race", "Ethnicity"]].describe(include="all")


# In[44]:


raw["Ethnicity"].value_counts(dropna=False)
raw["Race"].value_counts(dropna=False)
raw["Dizzy"].value_counts(dropna=False)


# In[45]:


codebook[
    codebook["Variable"].isin(["Ethnicity", "Race", "Dizzy"])
]


# In[46]:


clean = clean_data(raw, codebook)


# In[48]:


from clean import clean_data


# In[49]:


clean2 = clean_data(raw, codebook)


# In[50]:


clean2.equals(clean)


# EDA: Exploratory Data Analysis

# ### EDA Questions
# 
# 1. Which individual symptoms are associated with the highest rate of clinically important TBI (ciTBI)?
# 2. Which combinations of two or three symptoms are associated with the highest rate of ciTBI?

# In[51]:


clean["PosIntFinal"].value_counts(dropna=False)


# In[54]:


symptoms = [
    "LOCSeparate",
    "Seiz",
    "Vomit",
    "HA_verb",
    "AMS"
]


# In[53]:


[col for col in clean.columns if "Vom" in col or "HA" in col or "AMS" in col]


# In[55]:


for symptom in symptoms:
    print("\n", symptom)
    print(
        clean.groupby(symptom)["PosIntFinal"]
        .agg(["mean", "count"])
    )


# In[56]:


older = clean[clean["AgeinYears"] >= 2]

older.groupby("HA_verb")["PosIntFinal"].agg(["mean", "count"])


# In[57]:


headache = older[older["HA_verb"].isin([0, 1])]

headache.groupby("HA_verb")["PosIntFinal"].agg(["mean", "count"])


# question 2

# In[58]:


combo2 = (
    clean.groupby(["Seiz", "AMS"])["PosIntFinal"]
    .agg(["mean", "count"])
    .reset_index()
)

combo2


# In[59]:


combo2[
    (combo2["Seiz"] == 1) &
    (combo2["AMS"] == 1)
]


# In[60]:


from itertools import combinations


# In[61]:


pair_results = []

for s1, s2 in combinations(symptoms, 2):
    temp = clean[
        (clean[s1] == 1) &
        (clean[s2] == 1)
    ]

    rate = temp["PosIntFinal"].mean()
    count = temp["PosIntFinal"].count()

    pair_results.append({
        "symptom1": s1,
        "symptom2": s2,
        "ciTBI_rate": rate,
        "count": count
    })

pair_results = pd.DataFrame(pair_results)


# In[62]:


pair_results[
    pair_results["count"] >= 30
].sort_values(
    "ciTBI_rate",
    ascending=False
)


# 3 symptoms together

# In[63]:


triple_results = []

for s1, s2, s3 in combinations(symptoms, 3):
    temp = clean[
        (clean[s1] == 1) &
        (clean[s2] == 1) &
        (clean[s3] == 1)
    ]

    rate = temp["PosIntFinal"].mean()
    count = temp["PosIntFinal"].count()

    triple_results.append({
        "symptom1": s1,
        "symptom2": s2,
        "symptom3": s3,
        "ciTBI_rate": rate,
        "count": count
    })

triple_results = pd.DataFrame(triple_results)

triple_results[
    triple_results["count"] >= 30
].sort_values(
    "ciTBI_rate",
    ascending=False
)


# Among the symptom combinations examined, LOC + vomiting + altered mental status had the highest observed ciTBI rate (29.2%, n=325).

# making figures

# In[64]:


finding1 = []

for symptom in ["LOCSeparate", "Seiz", "Vomit", "AMS"]:
    temp = clean[clean[symptom] == 1]

    finding1.append({
        "Symptom": symptom,
        "ciTBI_rate": temp["PosIntFinal"].mean(),
        "count": temp["PosIntFinal"].count()
    })

# Headache: only patients age 2+
temp = clean[
    (clean["AgeinYears"] >= 2) &
    (clean["HA_verb"] == 1)
]

finding1.append({
    "Symptom": "HA_verb",
    "ciTBI_rate": temp["PosIntFinal"].mean(),
    "count": temp["PosIntFinal"].count()
})

finding1 = pd.DataFrame(finding1)
finding1


# In[65]:


finding1["Symptom"] = finding1["Symptom"].replace({
    "LOCSeparate": "Loss of consciousness",
    "Seiz": "Seizure",
    "Vomit": "Vomiting",
    "HA_verb": "Headache",
    "AMS": "Altered mental status"
})


# In[66]:


finding1 = finding1.sort_values(
    "ciTBI_rate",
    ascending=False
)


# In[67]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.bar(
    finding1["Symptom"],
    finding1["ciTBI_rate"] * 100
)

plt.ylabel("ciTBI rate (%)")
plt.xlabel("Symptom")
plt.title("Rate of Clinically Important TBI by Symptom")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# Seizure and altered mental status were associated with the highest observed ciTBI rates among the symptoms examined.

# Finding 2: symptom combinations

# In[68]:


top_pairs = (
    pair_results[pair_results["count"] >= 30]
    .sort_values("ciTBI_rate", ascending=False)
    .head(4)
    .copy()
)

top_pairs["Combination"] = (
    top_pairs["symptom1"] + " + " +
    top_pairs["symptom2"]
)


# In[69]:


top_triples = (
    triple_results[triple_results["count"] >= 30]
    .sort_values("ciTBI_rate", ascending=False)
    .head(4)
    .copy()
)

top_triples["Combination"] = (
    top_triples["symptom1"] + " + " +
    top_triples["symptom2"] + " + " +
    top_triples["symptom3"]
)


# In[70]:


finding2 = pd.concat([
    top_pairs[["Combination", "ciTBI_rate", "count"]],
    top_triples[["Combination", "ciTBI_rate", "count"]]
])

finding2 = finding2.sort_values(
    "ciTBI_rate",
    ascending=False
)


# In[71]:


finding2


# In[72]:


plt.figure(figsize=(9, 6))

plt.barh(
    finding2["Combination"],
    finding2["ciTBI_rate"] * 100
)

plt.xlabel("ciTBI rate (%)")
plt.ylabel("Symptom combination")
plt.title("ciTBI Rates for Selected Symptom Combinations")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[73]:


name_map = {
    "LOCSeparate": "Loss of consciousness",
    "Seiz": "Seizure",
    "Vomit": "Vomiting",
    "HA_verb": "Headache",
    "AMS": "Altered mental status"
}

finding2["Combination"] = finding2["Combination"].replace(
    name_map,
    regex=True
)


# In[74]:


plt.figure(figsize=(10, 6))

bars = plt.barh(
    finding2["Combination"],
    finding2["ciTBI_rate"] * 100
)

plt.xlabel("ciTBI rate (%)")
plt.ylabel("Symptom combination")
plt.title("ciTBI Rates for Selected Symptom Combinations")

for bar, n in zip(bars, finding2["count"]):
    plt.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"n={n}",
        va="center"
    )

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[75]:


plt.figure(figsize=(8, 5))

bars = plt.bar(
    finding1["Symptom"],
    finding1["ciTBI_rate"] * 100
)

plt.ylabel("ciTBI rate (%)")
plt.xlabel("Symptom")
plt.title("Rate of Clinically Important TBI by Symptom")

for bar, n in zip(bars, finding1["count"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"n={n}",
        ha="center"
    )

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ## Reality Check
# 
# The observed patterns are broadly consistent with the clinical context and the Kuppermann et al. study. Symptoms such as altered mental status and loss of consciousness were also important predictors of ciTBI in the original PECARN analysis. This provides some external support that the cleaned data preserve clinically meaningful relationships.

# ## stability check
# cutoff 30 -> 50

# In[76]:


stable_pairs = (
    pair_results[pair_results["count"] >= 50]
    .sort_values("ciTBI_rate", ascending=False)
    .head(4)
    .copy()
)

stable_triples = (
    triple_results[triple_results["count"] >= 50]
    .sort_values("ciTBI_rate", ascending=False)
    .head(4)
    .copy()
)


# In[77]:


stable_pairs["Combination"] = (
    stable_pairs["symptom1"] + " + " +
    stable_pairs["symptom2"]
)

stable_triples["Combination"] = (
    stable_triples["symptom1"] + " + " +
    stable_triples["symptom2"] + " + " +
    stable_triples["symptom3"]
)


# In[78]:


stable_finding2 = pd.concat([
    stable_pairs[["Combination", "ciTBI_rate", "count"]],
    stable_triples[["Combination", "ciTBI_rate", "count"]]
])

stable_finding2 = stable_finding2.sort_values(
    "ciTBI_rate",
    ascending=False
)


# In[79]:


stable_finding2["Combination"] = stable_finding2["Combination"].replace(
    name_map,
    regex=True
)


# In[80]:


stable_finding2


# before and after compariosn

# In[81]:


plt.figure(figsize=(10, 6))

plt.barh(
    finding2["Combination"],
    finding2["ciTBI_rate"] * 100
)

plt.xlabel("ciTBI rate (%)")
plt.ylabel("Symptom combination")
plt.title("Minimum Cell Size = 30")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[82]:


plt.figure(figsize=(10, 6))

plt.barh(
    stable_finding2["Combination"],
    stable_finding2["ciTBI_rate"] * 100
)

plt.xlabel("ciTBI rate (%)")
plt.ylabel("Symptom combination")
plt.title("Minimum Cell Size = 50")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Increasing the minimum cell size from 30 to 50 removed some smaller symptom combinations, but the highest-risk combinations remained largely unchanged. This suggests that the main finding is reasonably stable to this threshold choice.

# ## Part 2: Modeling
# 
# I use clinically available pre-CT characteristics to predict clinically important TBI (ciTBI).
# I compare logistic regression with a decision tree.

# In[83]:


model_data = clean[
    [
        "AgeinYears",
        "GCSTotal",
        "LOCSeparate",
        "Seiz",
        "Vomit",
        "AMS",
        "PosIntFinal"
    ]
].copy()


# In[84]:


model_data = model_data.dropna(subset=["PosIntFinal"])


# In[85]:


model_data["PosIntFinal"].value_counts()


# In[86]:


X = model_data.drop(columns=["PosIntFinal"])
y = model_data["PosIntFinal"]


# In[87]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[88]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)


# In[89]:


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

logit.fit(X_train_imp, y_train)


# In[90]:


logit_pred = logit.predict(X_test_imp)


# In[91]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=4,
    class_weight="balanced",
    random_state=42
)

tree.fit(X_train_imp, y_train)


# In[92]:


tree_pred = tree.predict(X_test_imp)


# In[93]:


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score
)


# In[94]:


print("Logistic Regression")
print(confusion_matrix(y_test, logit_pred))
print(classification_report(y_test, logit_pred))

print(
    "Sensitivity:",
    recall_score(y_test, logit_pred)
)


# In[95]:


print("Decision Tree")
print(confusion_matrix(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

print(
    "Sensitivity:",
    recall_score(y_test, tree_pred)
)


# In[96]:


logit_coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logit.coef_[0]
})

logit_coef.sort_values("Coefficient", ascending=False)


# In[97]:


tree_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree.feature_importances_
})

tree_importance.sort_values("Importance", ascending=False)


# In[98]:


from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8))

plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No ciTBI", "ciTBI"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Predicting ciTBI")
plt.show()


# ### Model Interpretation
# 
# The logistic regression model had higher sensitivity than the decision tree, identifying about 78% of ciTBI cases compared with about 71% for the decision tree. However, the logistic regression also produced more false positives. Because the main clinical goal is to avoid missing serious brain injuries, sensitivity is especially important in this setting.
# 
# In the logistic regression, altered mental status, vomiting, loss of consciousness, and seizure were positively associated with ciTBI, while a higher GCS score was negatively associated with ciTBI. The decision tree also relied heavily on altered mental status, with GCS and loss of consciousness playing smaller roles.
# 
# The logistic regression is useful because the direction of each association is easy to interpret, while the decision tree is useful because its prediction process can be represented as a simple sequence of clinical rules. Both models suggest that altered mental status is an important indicator of ciTBI risk.

# ## Discussion
# 
# The dataset was large but manageable. The main challenge was data cleaning because many variables used special codes such as 92 for "not applicable." The documentation was important for understanding which values were true missing values and which had specific meanings.
# 
# This lab also shows the three realms of data science. The original clinical setting represents data and reality, the logistic regression and decision tree represent models, and applying these models to future patients represents future data and reality.
# 
# The data do not perfectly represent reality. Some symptoms are difficult to measure, some values are missing, and CT decisions are not random. For this reason, the associations in this analysis should not be interpreted as causal.
# 
# Overall, altered mental status, seizure, and loss of consciousness were associated with higher ciTBI rates. The logistic regression had higher sensitivity than the decision tree, but it also produced more false positives. This reflects the tradeoff between detecting serious injuries and avoiding unnecessary CT scans.
