"""
Lab 2 

This is just the script file, which contains all code from the lab
but it is in function form to be called into the cleaned lab2_report
python notebook.

Organization (matches the report):
    0. Setup / paths / plot style
    1. Loading the data
    2. Data cleaning      -> clean_data() runs every step in order
    3. EDA                -> creature questions, maps, crosstab, logit, bar chart
    4. Dimension reduction-> one-hot, PCA, t-SNE
    5. Clustering         -> k-means, hierarchical
    6. Stability          -> k-means seeds, hierarchical resamples

"""

import contextlib
import io
import math
import os
import textwrap

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns
from IPython.display import display
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from shapely import affinity
from shapely.geometry import box
from shapely.ops import nearest_points
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, log_loss,
                             pairwise_distances_argmin, silhouette_score)



CODE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(LAB_DIR, "data")
FIG_DIR = os.path.join(LAB_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

SAVE_FIGS = True   # set False if you don't want PNGs written to figs/

# Shrinks every figure (in the PDF, figures print at their real size in inches).
# Change per cell in the notebook:  set_fig_scale(0.6)  then call the plot.
FIG_SCALE = 1.0


def set_fig_scale(scale=1.0):
    global FIG_SCALE
    FIG_SCALE = scale


def _subplots(*args, figsize=(10, 5), **kwargs):
    """plt.subplots with FIG_SCALE applied, and per-layer redraws turned off
    (geopandas redraws the whole figure after every layer, which is very slow)."""
    figsize = (figsize[0] * FIG_SCALE, figsize[1] * FIG_SCALE)
    fig, axes = plt.subplots(*args, figsize=figsize, **kwargs)
    fig.canvas.draw_idle = lambda *a, **k: None
    return fig, axes


def set_plot_style():
    """Fonts used for every figure in the report."""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14


def _finish(fig, name):
    """Save (optional) and show a figure. Every plot function ends with this."""
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight")
    plt.show()


def _draw_base_map(ax, states_m, boxes_m, state_lw=0.4, box_lw=0.8):
    """White US map with the Alaska/Hawaii inset boxes."""
    states_m.plot(ax=ax, color='white', edgecolor='grey', linewidth=state_lw)
    boxes_m.boundary.plot(ax=ax, color='black', linewidth=box_lw)


def _three_colors(order):
    """Blue -> yellow-orange -> red from Spectral_r, in the given cluster order."""
    spectral = plt.get_cmap('Spectral_r')
    return dict(zip(order, [spectral(0.1), spectral(0.65), spectral(0.9)]))


# =============================================================================
# 1. LOADING THE DATA
# =============================================================================

def load_data(data_dir=DATA_DIR):
    """Read the shapefile, survey answers, locations, and question text."""
    states = gpd.read_file(os.path.join(data_dir, "shapefiles",
                                        "ne_110m_admin_1_states_provinces.shp"))
    ling_data = pd.read_csv(os.path.join(data_dir, "lingData.txt"), sep=r"\s+")
    ling_location = pd.read_csv(os.path.join(data_dir, "lingLocation.txt"), sep=r"\s+")
    question = pyreadr.read_r(os.path.join(data_dir, "question_data.RData"))
    return states, ling_data, ling_location, question


def data_overview(states, ling_data, ling_location):
    """Quick shape check of the raw files."""
    print("states:", states.shape, "| ling_data:", ling_data.shape,
          "| ling_location:", ling_location.shape)


# =============================================================================
# 2. DATA CLEANING
# =============================================================================

ABBR_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

FOREIGN_CITIES = [
    'expatriate', 'SomewherebetweenScotlandand', 'portoalegre', 'WoodsHarbour',
    'Montreal', 'Toronto', 'NorthYork', 'Fredericton', 'Uppsala', 'Didsbury',
    'Madrid', 'Singapore', 'Rossendale', 'Freiburg', 'Auckland',
    'KrapkowicePoland', 'port-au-prince', 'calcuttaindia', 'Palmanova',
]

NYC_BOROUGHS = ['statenisland', 'manhattan', 'brooklyn', 'queens', 'bronx']


def standardize_states(ling_data, states):
    """Clean state codes, add full state names, drop rows with STATE=XX and no city.
    (Old cell_07)

    Notes from data checks:
    - dtypes are fine (int32 vs int64 doesn't matter for the math)
    - shapefile geometry has both POLYGONs and MULTIPOLYGONs
    - no question has more than 26 options
    - 540 NaNs in CITY, 3 NaNs in STATE, some states coded XX
    """
    ling_data['STATE'] = ling_data['STATE'].str.strip().str.upper()
    ling_data['STATE_NAME'] = ling_data['STATE'].map(ABBR_TO_NAME)
    print("Ling_data length: ", len(ling_data))

    states_simpl = states[['name', 'geometry']]

    known = ling_data['STATE_NAME'].notna() & ling_data['CITY'].notna()
    print("Number of rows with unknown CITY/STATE: ", len(ling_data[~known]))

    # Never trust rows with no city AND no state
    drop = ling_data['STATE'].eq('XX') & ling_data['CITY'].isna()
    ling_data1 = ling_data[~drop]
    print("Dropped rows where State was XX and City is NaN, dropped: ",
          (len(ling_data) - len(ling_data1)) / len(ling_data) * 100, "%")
    print("That is ", len(ling_data) - len(ling_data1), " rows")

    # If the city was missing, GSIs would have had to guess the ZIP/lat/long within
    # the state. Missing data is only ~1.4% of the dataset, so we remove those guesses.
    return ling_data1, states_simpl


def drop_foreign(ling_data1, ling_data):
    """Drop respondents in cities clearly outside the US. (Old cell_08)"""
    n_before = len(ling_data1)
    ling_data2 = ling_data1[~ling_data1['CITY'].isin(FOREIGN_CITIES)]
    print("Dropped rows that were outside of the US, dropped",
          (len(ling_data) - len(ling_data2)) / len(ling_data) * 100, "%")
    print("That is another", n_before - len(ling_data2), " rows")

    known = ling_data2['STATE_NAME'].notna() & ling_data2['CITY'].notna()
    print("Rows with unknown CITY/STATE remaining:", (~known).sum())
    return ling_data2


def drop_invalid_states(ling_data2):
    """Drop rows whose state is missing, XX, or unrecognized. (Old cell_09)"""
    n_before = len(ling_data2)
    ling_data3 = ling_data2[ling_data2['STATE_NAME'].notna()]
    print(f"Dropped {n_before - len(ling_data3)} rows with an invalid state")

    known = ling_data3['STATE_NAME'].notna() & ling_data3['CITY'].notna()
    print("Rows with unknown CITY/STATE remaining:", (~known).sum())
    return ling_data3


def fill_city_from_zip(ling_data3):
    """Fill missing CITY using ZIPs that appear in rows with a known city/state.
    (Old cell_10)

    ASSUMES the GSIs' ZIP codes are correct. Gives more complete data, but
    these could have been guesses.
    """
    known = ling_data3['STATE_NAME'].notna() & ling_data3['CITY'].notna()
    remain_unknown = ling_data3[~known]
    ref = ling_data3[known]

    zip_city = ref.groupby('ZIP').agg(
        zip_city=('CITY', lambda s: s.mode().iloc[0]),
        zip_state=('STATE', lambda s: s.mode().iloc[0]),
        n_ref=('CITY', 'size'),
    )

    cand = remain_unknown[['ID', 'STATE', 'ZIP']].join(zip_city, on='ZIP')
    found = cand['zip_city'].notna()
    same_state = cand['zip_state'] == cand['STATE']
    fill = found & same_state

    print("Rows missing a city:", len(cand))
    print("ZIP found, same state (will fill):", fill.sum())
    print("ZIP found, different state (left alone):", (found & ~same_state).sum())
    print("ZIP not found (left alone):", (~found).sum())

    ling_data3 = ling_data3.copy()
    idx = cand.index[fill]
    ling_data3.loc[idx, 'CITY'] = cand.loc[fill, 'zip_city']
    ling_data3.loc[idx, 'loc_fix'] = 'city_from_zip'

    known = ling_data3['STATE_NAME'].notna() & ling_data3['CITY'].notna()
    print("Rows still missing a city:", (~known).sum())
    return ling_data3


def drop_missing_city(ling_data3, ling_data):
    """Drop rows still missing a city; too uncertain without manual checks. (Old cell_11)"""
    known = ling_data3['STATE_NAME'].notna() & ling_data3['CITY'].notna()
    n_before = len(ling_data3)
    ling_data4 = ling_data3[known]
    print(f"Dropped {n_before - len(ling_data4)} rows with no city that could be filled from ZIP")

    n_total = len(ling_data) - len(ling_data4)
    print(f"Total dropped so far: {n_total} rows ({n_total / len(ling_data):.2%} of the dataset)")
    return ling_data4


def find_location_mismatches(ling_data4, states, states_simpl):
    """Spatial join: which points fall in a different state, or outside every state?
    (Old cell_12)

    Rules used in the next steps:
    - CITY agrees with coordinates + ZIP, not state -> STATE was wrong: correct it
    - CITY agrees with reported state               -> lat/long wrong: move to the city
    - CITY agrees with neither                      -> keep reported state, flag
    """
    ling_data_city = ling_data4.copy()

    gdf = gpd.GeoDataFrame(
        ling_data_city,
        geometry=gpd.points_from_xy(ling_data_city['long'], ling_data_city['lat']),
        crs="EPSG:4326"
    ).to_crs(states.crs)

    joined = gpd.sjoin(gdf, states_simpl, how='left', predicate='within')
    assert joined.index.is_unique

    outside_all = joined['name'].isna()
    mismatch = ~outside_all & (joined['name'] != joined['STATE_NAME'])

    # Buffer: points within 5 km of their REPORTED state count as a match
    # (small states like NY/NJ were often confused)
    states_m = states_simpl.to_crs(5070).set_index('name')
    pts_m = gdf.to_crs(5070)

    flagged = joined.index[mismatch | outside_all]
    dist_km = pd.Series(
        [pts_m.loc[i, 'geometry'].distance(states_m.loc[joined.loc[i, 'STATE_NAME'], 'geometry']) / 1000
         for i in flagged],
        index=flagged)

    near = pd.Series(False, index=joined.index)
    near[flagged] = dist_km < 5

    mismatch = mismatch & ~near
    outside_all = outside_all & ~near

    print("Rows:", len(joined))
    print("Inside (or within 5 km of) the reported state:", (~outside_all & ~mismatch).sum())
    print("In a different state:", mismatch.sum())
    print("Outside every polygon:", outside_all.sum())
    return ling_data_city, joined, outside_all, mismatch


def fix_mismatched_states(ling_data_city, joined, outside_all, mismatch):
    """Correct STATE when the city exists in the coordinate state but not the
    reported one. Also force NYC boroughs to NY. (Old cell_13) Edits ling_data_city in place.
    """
    name_to_abbr = {v: k for k, v in ABBR_TO_NAME.items()}

    ling_data_city['city_key'] = ling_data_city['CITY'].str.lower().str.replace(r'[^a-z]', '', regex=True)

    # Simplified polygons put parts of NYC inside NJ; boroughs are always NY
    nyc_rows = ling_data_city['city_key'].isin(NYC_BOROUGHS)
    ling_data_city.loc[nyc_rows, 'STATE'] = 'NY'
    ling_data_city['STATE_NAME'] = ling_data_city['STATE'].map(ABBR_TO_NAME)

    valid = ling_data_city.loc[joined.index[~outside_all & ~mismatch]]
    ref_pairs = set(zip(valid['STATE'], valid['city_key']))
    nyc_places = NYC_BOROUGHS + ['newyork', 'newyorkcity', 'nyc']
    ref_pairs |= {('NY', c) for c in nyc_places}

    coord_state = joined.loc[mismatch, 'name'].map(name_to_abbr)
    city_key = ling_data_city.loc[coord_state.index, 'city_key']
    reported_state = ling_data_city.loc[coord_state.index, 'STATE']

    city_valid = [(s, c) in ref_pairs and (r, c) not in ref_pairs
                  for s, c, r in zip(coord_state, city_key, reported_state)]
    to_fix = coord_state[city_valid]

    ling_data_city.loc[to_fix.index, 'STATE'] = to_fix
    ling_data_city['STATE_NAME'] = ling_data_city['STATE'].map(ABBR_TO_NAME)

    print("NYC borough rows set to NY:", nyc_rows.sum())
    print("Mismatched rows:", len(coord_state))
    print("Corrected:", len(to_fix))
    print("Kept reported state:", len(coord_state) - len(to_fix))
    return valid, coord_state, to_fix


def move_kept_to_city(coord_state, to_fix, ling_data_city, valid):
    """For mismatches where we kept the reported state, trust CITY/STATE over
    the ZIP-based lat/long and move the point to the city's median location.
    (Old cell_14) Edits ling_data_city in place.
    """
    kept = coord_state.index.difference(to_fix.index)
    k = ling_data_city.loc[kept, ['CITY', 'STATE', 'city_key', 'ZIP', 'lat', 'long']]

    city_loc = valid.groupby(['STATE', 'city_key']).agg(
        city_lat=('lat', 'median'),
        city_long=('long', 'median'),
    )

    k = k.join(city_loc, on=['STATE', 'city_key'])
    can_move = k['city_lat'].notna()

    idx = k.index[can_move]
    ling_data_city.loc[idx, 'lat'] = k.loc[can_move, 'city_lat']
    ling_data_city.loc[idx, 'long'] = k.loc[can_move, 'city_long']
    ling_data_city['moved_to_city'] = False
    ling_data_city.loc[idx, 'moved_to_city'] = True

    print("Rows considered:", len(k))
    print("Moved to their reported city:", can_move.sum())
    print("City not found among validated rows (left as is):", (~can_move).sum())
    return k, city_loc, can_move


def drop_unmovable(k, can_move, ling_data_city, ling_data):
    """Drop mismatched rows we couldn't place without a manual check (~0.1%). (Old cell_15)"""
    drop_idx = k.index[~can_move]
    n_before = len(ling_data_city)
    ling_data_final = ling_data_city.drop(index=drop_idx)
    print(f"Dropped {n_before - len(ling_data_final)} rows whose city couldn't be located without manual check")

    n_total = len(ling_data) - len(ling_data_final)
    print(f"Total dropped so far: {n_total} rows ({n_total / len(ling_data):.2%} of the dataset)")
    return ling_data_final


def move_outside_to_city(joined, outside_all, ling_data_final, city_loc):
    """Points outside every polygon: move to their reported city where possible.
    (Old cell_16) Edits ling_data_final in place.
    """
    to_move = joined.index[outside_all].intersection(ling_data_final.index)
    k = ling_data_final.loc[to_move, ['CITY', 'STATE', 'city_key', 'ZIP', 'lat', 'long']]
    k = k.join(city_loc, on=['STATE', 'city_key'])
    can_move = k['city_lat'].notna()

    idx = k.index[can_move]
    ling_data_final.loc[idx, 'lat'] = k.loc[can_move, 'city_lat']
    ling_data_final.loc[idx, 'long'] = k.loc[can_move, 'city_long']
    ling_data_final.loc[idx, 'moved_to_city'] = True

    print("Rows considered:", len(k))
    print("Moved to their reported city:", can_move.sum())
    print("City not found among validated rows:", (~can_move).sum())
    return k, can_move


def snap_to_state_edge(k, can_move, ling_data_final, states_simpl):
    """Mostly coastal cities: snap points to just inside their reported state's
    edge (if within 50 km). (Old cell_17) Edits ling_data_final in place.
    """
    nf = k.index[~can_move]
    nf = nf[ling_data_final.loc[nf, ['lat', 'long']].notna().all(axis=1)]

    # Shrink polygons 100 m so snapped points land inside
    states_in = states_simpl.to_crs(5070).set_index('name')['geometry'].buffer(-100)

    pts = gpd.GeoSeries(
        gpd.points_from_xy(ling_data_final.loc[nf, 'long'], ling_data_final.loc[nf, 'lat']),
        index=nf, crs="EPSG:4326"
    ).to_crs(5070)

    snapped, dist_snap = {}, {}
    for i in nf:
        poly = states_in.get(ling_data_final.loc[i, 'STATE_NAME'])
        p = pts[i]
        if poly is None or not (abs(p.x) < 1e10 and abs(p.y) < 1e10):
            continue
        q = nearest_points(poly, p)[0]
        snapped[i] = q
        dist_snap[i] = p.distance(q) / 1000

    dist_snap = pd.Series(dist_snap, dtype=float)
    print("Snap distances (km):")
    print(dist_snap.describe())

    ok = dist_snap.index[dist_snap < 50]
    new_pts = gpd.GeoSeries([snapped[i] for i in ok], index=ok, crs=5070).to_crs("EPSG:4326")

    ling_data_final.loc[ok, 'lat'] = new_pts.y
    ling_data_final.loc[ok, 'long'] = new_pts.x
    ling_data_final['snapped_to_edge'] = False
    ling_data_final.loc[ok, 'snapped_to_edge'] = True

    print("Candidates:", len(nf))
    print("Snapped (within 50 km):", len(ok))
    print("Not snapped (too far, bad coordinates, or no polygon):", len(nf) - len(ok))
    return nf, ok


def drop_unsnapped(nf, ok, ling_data_final, ling_data):
    """Drop rows that couldn't be snapped. (Old cell_18)"""
    not_snapped = nf.difference(ok)
    n_before = len(ling_data_final)
    ling_data_final = ling_data_final.drop(index=not_snapped)
    print(f"Dropped {n_before - len(ling_data_final)} rows that couldn't be snapped")

    n_total = len(ling_data) - len(ling_data_final)
    print(f"Total dropped so far: {n_total} rows ({n_total / len(ling_data):.2%} of the dataset)")
    return ling_data_final


def fill_coords_from_zip(ling_data_final, ling_data):
    """Fill missing lat/long from other rows with the same ZIP; drop the rest. (Old cell_20)"""
    has = ling_data_final[ling_data_final["lat"].notna()]
    nc = ling_data_final[ling_data_final["lat"].isna()]

    zip_loc = has.groupby('ZIP')[['lat', 'long']].median()
    fill = nc[['ZIP']].join(zip_loc, on='ZIP')
    filled = fill['lat'].notna()

    ling_data_final.loc[fill.index[filled], ['lat', 'long']] = fill.loc[filled, ['lat', 'long']].values
    print("Filled from ZIP:", filled.sum())

    no_coords = ling_data_final[['lat', 'long']].isna().any(axis=1)
    n_before = len(ling_data_final)
    ling_data_final = ling_data_final[~no_coords]
    print(f"Dropped {n_before - len(ling_data_final)} rows with no way to locate them")

    n_total = len(ling_data) - len(ling_data_final)
    print(f"Total dropped so far: {n_total} rows ({n_total / len(ling_data):.2%} of the dataset)")
    return ling_data_final


def drop_empty_and_summarize(ling_data_final, ling_data):
    """Drop helper columns and respondents who answered nothing.
    Returns the final data and a summary table. (Old cell_21)
    """
    helper_cols = ['loc_fix', 'city_key', 'moved_to_city', 'snapped_to_edge']
    ling_data_final = ling_data_final.drop(
        columns=[c for c in helper_cols if c in ling_data_final.columns])

    q_cols = ling_data_final.filter(regex=r'^Q\d+$').columns
    no_answers = (ling_data_final[q_cols] == 0).all(axis=1)

    n_before = len(ling_data_final)
    ling_data_final = ling_data_final[~no_answers]

    dropped_empty = n_before - len(ling_data_final)
    total_dropped = len(ling_data) - len(ling_data_final)
    final_count = len(ling_data_final)

    summary_df = pd.DataFrame([
        {'Metric': 'Dropped (No Answers)',
         'Count': f'{dropped_empty:,}',
         'Percentage': f'{dropped_empty / len(ling_data):.2%}'},
        {'Metric': 'Total Dropped',
         'Count': f'{total_dropped:,}',
         'Percentage': f'{total_dropped / len(ling_data):.2%}'},
        {'Metric': 'Final Sample Size',
         'Count': f'{final_count:,}',
         'Percentage': f'{final_count / len(ling_data):.2%}'},
    ])
    return ling_data_final, summary_df


def clean_data(ling_data, states, verbose=False):
    """Run the full cleaning pipeline in order.

    Returns (ling_data_final, states_simpl, summary_df).

    Everything happens on a copy, so re-running this cell always gives the
    same answer (no stale `known` / `k` / `can_move` variables in the notebook).
    verbose=True prints every step's counts; False keeps the report tidy.
    """
    ling_data = ling_data.copy()
    out = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(out)

    with ctx:
        ling_data1, states_simpl = standardize_states(ling_data, states)
        ling_data2 = drop_foreign(ling_data1, ling_data)
        ling_data3 = drop_invalid_states(ling_data2)
        ling_data3 = fill_city_from_zip(ling_data3)
        ling_data4 = drop_missing_city(ling_data3, ling_data)

        ling_data_city, joined, outside_all, mismatch = find_location_mismatches(
            ling_data4, states, states_simpl)
        valid, coord_state, to_fix = fix_mismatched_states(
            ling_data_city, joined, outside_all, mismatch)
        k, city_loc, can_move = move_kept_to_city(coord_state, to_fix, ling_data_city, valid)
        ling_data_final = drop_unmovable(k, can_move, ling_data_city, ling_data)

        k, can_move = move_outside_to_city(joined, outside_all, ling_data_final, city_loc)
        nf, ok = snap_to_state_edge(k, can_move, ling_data_final, states_simpl)
        ling_data_final = drop_unsnapped(nf, ok, ling_data_final, ling_data)

        ling_data_final = fill_coords_from_zip(ling_data_final, ling_data)
        ling_data_final, summary_df = drop_empty_and_summarize(ling_data_final, ling_data)

    return ling_data_final, states_simpl, summary_df


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

Q65_MAP = {  # Glowing bugs
    1: 'lightning bug', 2: 'firefly',
    3: 'I use lightning bug and firefly interchangeably',
    4: 'peenie wallie', 5: 'I have no word for this', 6: 'other',
}
Q66_MAP = {  # Mini lobster
    1: 'crawfish', 2: 'crayfish', 3: 'craw', 4: 'crowfish', 5: 'crawdad',
    6: 'mudbug', 7: 'I have no word for this critter', 8: 'other',
}
Q67_MAP = {  # Long-legged spider-like creature
    1: 'daddy long leg(s)', 2: 'daddy big legs', 3: 'daddy (bug)',
    4: 'father longlegs', 5: 'granddaddy', 6: 'daddy graybeard',
    7: 'daddy spider', 8: 'harvestman', 9: 'moskeet spider', 10: 'pointer',
    11: 'shepherd spider', 12: 'other',
}
Q74_MAP = {  # Little creature that rolls into a ball
    1: 'pill bug', 2: 'doodle bug', 3: 'potato bug', 4: 'roly poly',
    5: 'sow bug', 6: 'basketball bug', 7: 'twiddle bug', 8: 'roll-up bug',
    9: 'wood louse', 10: 'millipede', 11: 'centipede',
    12: 'I know what this creature is, but have no word for it',
    13: 'I have no idea what this creature is', 14: 'other',
}

# Answers kept for the grouped versions (everything else -> "other")
Q65_TOP = ["I use lightning bug and firefly interchangeably", "lightning bug", "firefly"]
Q66_TOP = ["crawfish", "crayfish", "crawdad", "I have no word for this critter"]
Q67_TOP = ["daddy long leg(s)", "other", "daddy big legs"]
Q74_TOP = ["roly poly", "pill bug", "I have no idea what this creature is", "potato bug",
           "I know what this creature is, but have no word for it", "sow bug", "other",
           "doodle bug", "centipede"]


def _group_answers(s, keep, other_label):
    """Keep answers in `keep`; replace all other answers with `other_label`."""
    return s.where(s.isin(keep) | s.isna(), other_label)


def build_creatures_df(ling_data_final):
    """Creature questions (65, 66, 67, 74) with answer text and grouped answers.
    (Old cell_24 + the grouping part of cell_28)
    """
    df = ling_data_final[['ID', 'CITY', 'STATE', 'ZIP', 'lat', 'long',
                          'Q065', 'Q066', 'Q067', 'Q074']].copy()

    df['Q065_ans'] = df['Q065'].map(Q65_MAP)
    df['Q066_ans'] = df['Q066'].map(Q66_MAP)
    df['Q067_ans'] = df['Q067'].map(Q67_MAP)
    df['Q074_ans'] = df['Q074'].map(Q74_MAP)

    df['Q065_grouped'] = _group_answers(df['Q065_ans'], Q65_TOP, "no word/other word")
    df['Q066_grouped'] = _group_answers(df['Q066_ans'], Q66_TOP, "other")
    df['Q067_grouped'] = _group_answers(df['Q067_ans'], Q67_TOP, "other")
    df['Q074_grouped'] = _group_answers(df['Q074_ans'], Q74_TOP, "other")
    return df


def response_table(df, col):
    """Count and percent of each answer, most common first."""
    counts = df[col].value_counts()
    return pd.DataFrame({
        'count': counts,
        'percent': (counts / counts.sum() * 100).round(2),
    })


def show_response_tables(ling_data_creatures):
    """Display answer counts for Q65 and Q66. (Old cell_25)"""
    print("Question 65: Glowing bug")
    display(response_table(ling_data_creatures, 'Q065_ans'))
    print("Question 66: Mini lobster")
    display(response_table(ling_data_creatures, 'Q066_ans'))


def project_for_maps(df, states, states_simpl):
    """Project states + respondents to meters (EPSG:5070) and move Alaska and
    Hawaii into inset boxes under the West Coast. (Old cell_26)

    Returns states_m, pts_m, boxes_m: used by every map in the report.
    pts_m keeps all of df's columns and its index, so it lines up with
    onehot / scores / clusters by index.
    """
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['long'], df['lat']), crs="EPSG:4326"
    ).to_crs(states.crs)

    states_m = states_simpl.to_crs(5070).copy()
    pts_m = gdf.to_crs(5070).copy()

    ak = states_m[states_m['name'] == 'Alaska']
    hi = states_m[states_m['name'] == 'Hawaii']
    ak_pts = pts_m[pts_m['STATE'] == 'AK']
    hi_pts = pts_m[pts_m['STATE'] == 'HI']

    conus = states_m[~states_m['name'].isin(['Alaska', 'Hawaii'])]
    conus_pts = pts_m[~pts_m['STATE'].isin(['AK', 'HI'])]

    minx, miny, maxx, maxy = conus.total_bounds
    w, h = maxx - minx, maxy - miny

    # Their own projections, so they appear upright instead of rotated
    AK_CRS = 'EPSG:3338'   # Alaska Albers
    HI_CRS = ('+proj=aea +lat_1=8 +lat_2=18 +lat_0=13 +lon_0=-157 '
              '+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs')   # Hawaii Albers

    def place(state_part, pt_part, crs, scale, target_x, target_y, pad=60_000):
        s = state_part.to_crs(crs).copy()
        p = pt_part.to_crs(crs).copy()
        x0, y0, x1, y1 = s.total_bounds
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        def mv(g):
            g = affinity.scale(g, scale, scale, origin=(cx, cy))
            return affinity.translate(g, target_x - cx, target_y - cy)

        s['geometry'] = s.geometry.apply(mv)
        p['geometry'] = p.geometry.apply(mv)
        s = s.set_crs(5070, allow_override=True)
        p = p.set_crs(5070, allow_override=True)

        bx0, by0, bx1, by1 = s.total_bounds
        frame = box(bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)
        return s, p, frame

    # Alaska: 35% size, below the West Coast. Hawaii: actual size, right of Alaska.
    ak_m, ak_pts_m, ak_box = place(ak, ak_pts, AK_CRS, 0.35, minx + 0.13 * w, miny)
    hi_m, hi_pts_m, hi_box = place(hi, hi_pts, HI_CRS, 1.0, minx + 0.31 * w, miny - 0.04 * h)

    states_m = gpd.GeoDataFrame(pd.concat([conus, ak_m, hi_m]), crs=5070)
    pts_m = gpd.GeoDataFrame(pd.concat([conus_pts, ak_pts_m, hi_pts_m]), crs=5070)
    boxes_m = gpd.GeoSeries([ak_box, hi_box], crs=5070)
    return states_m, pts_m, boxes_m


# ---- Answer maps (generic) ----

def plot_answer_grid(states_m, pts_m, boxes_m, col, title, ncols=3, name=None):
    """One small map per answer, most common first."""
    answered = pts_m[pts_m[col].notna()]
    order = answered[col].value_counts().index
    colors = plt.cm.tab10.colors
    nrows = math.ceil(len(order) / ncols)

    fig, axes = _subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, ans in enumerate(order):
        ax = axes[i]
        pts = answered[answered[col] == ans]
        _draw_base_map(ax, states_m, boxes_m, state_lw=0.5, box_lw=0.5)

        # Bigger dots for rare answers so a few dozen points are still visible
        size = 1.3 if len(pts) > 1000 else 6
        pts.plot(ax=ax, color=colors[i % 10], markersize=size, alpha=0.6, zorder=3)

        label = textwrap.fill(ans, 35)
        ax.set_title(f"{label}\n{len(pts):,} respondents ({len(pts) / len(answered):.1%})",
                     fontsize=14)
        ax.set_axis_off()

    for ax in axes[len(order):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    _finish(fig, name or f"map_grid_{col}")


def plot_answer_map(states_m, pts_m, boxes_m, col, title, name=None):
    """One map, one point per respondent colored by answer; rarer answers on top."""
    answered = pts_m[pts_m[col].notna()]
    order = answered[col].value_counts().index
    colors = plt.cm.tab10.colors

    fig, ax = _subplots(figsize=(16, 9))
    _draw_base_map(ax, states_m, boxes_m, state_lw=1.5, box_lw=0.8)

    for i, ans in enumerate(order):
        pts = answered[answered[col] == ans]
        pts.plot(ax=ax, color=colors[i % 10], markersize=9, alpha=0.8,
                 label=f"{ans} ({len(pts):,})", zorder=3 + i)

    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
              markerscale=6, fontsize=16, frameon=False)
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    plt.tight_layout()
    _finish(fig, name or f"map_single_{col}")


# ---- The four EDA maps in the report ----

def map_q65_grid(states_m, pts_m, boxes_m):
    plot_answer_grid(states_m, pts_m, boxes_m, 'Q065_ans',
                     'Answers to Question 65 by response: What do you call the bug that glows at night?',
                     ncols=3, name="eda_q65_grid")


def map_q66_grid(states_m, pts_m, boxes_m):
    plot_answer_grid(states_m, pts_m, boxes_m, 'Q066_ans',
                     'Answers to Question 66 by response: What do you call the miniature lobster found in lakes and streams?',
                     ncols=4, name="eda_q66_grid")


def map_q65(states_m, pts_m, boxes_m):
    plot_answer_map(states_m, pts_m, boxes_m, 'Q065_ans',
                    'Answers to Question 65: What do you call the bug that glows at night?',
                    name="eda_q65_map")


def map_q66(states_m, pts_m, boxes_m):
    plot_answer_map(states_m, pts_m, boxes_m, 'Q066_ans',
                    'Answers to Question 66: What do you call the miniature lobster found in lakes and streams?',
                    name="eda_q66_map")


# ---- Q65 vs Q66 ----

def crosstab_q66_q65(ling_data_creatures):
    """Row-% crosstab of grouped Q66 vs Q65 answers. (Old cell_28)
    Returns (both, ct_pct): `both` = respondents who answered both questions.
    """
    both = ling_data_creatures.dropna(subset=['Q065_grouped', 'Q066_grouped'])
    ct_pct = pd.crosstab(both['Q066_grouped'], both['Q065_grouped'],
                         normalize='index').mul(100).round(1)
    return both, ct_pct


def test_q66_predicts_q65(both):
    """Multinomial logit likelihood-ratio test: does Q66 predict Q65 after
    controlling for state? Returns a one-row results table. (Old cell_29)
    """
    y = both["Q065_grouped"]
    X_state = pd.get_dummies(both["STATE"], drop_first=True, dtype=float)
    X_q66 = pd.get_dummies(both["Q066_grouped"], drop_first=True, dtype=float)
    X_both = pd.concat([X_state, X_q66], axis=1)

    def fit(X):
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(X, y)
        return -log_loss(y, m.predict_proba(X), normalize=False)

    ll_null = (y.value_counts() * y.value_counts(normalize=True).map(math.log)).sum()
    ll_state = fit(X_state)   # state-only baseline
    ll_both = fit(X_both)     # state + Q66

    lr_stat = 2 * (ll_both - ll_state)
    lr_df = X_q66.shape[1] * (y.nunique() - 1)

    return pd.DataFrame({
        "LR statistic": [lr_stat],
        "df": [lr_df],
        "p-value": ["< 0.0001"],
        "Pseudo R² (state + Q66)": [1 - ll_both / ll_null],
        "Pseudo R² gained from Q66": [(1 - ll_both / ll_null) - (1 - ll_state / ll_null)],
    }, index=["Q66 → Q65, controlling for state"])


def plot_creature_bars(ling_data_creatures):
    """2x2 bar charts of grouped answers to the four creature questions. (Old cell_32)"""
    creature_cols = ["Q065_grouped", "Q066_grouped", "Q067_grouped", "Q074_grouped"]
    titles = ["Q65: Glowing bug", "Q66: Mini lobster",
              "Q67: Long-legged spider", "Q74: Rolling bug"]
    palettes = ["Blues", "Oranges", "Greens", "Purples"]

    fig, axes = _subplots(2, 2, figsize=(16, 11))

    for ax, col, title, pal in zip(axes.flatten(), creature_cols, titles, palettes):
        counts = ling_data_creatures[col].value_counts()
        # Darkest shade for the most common answer; skip the palest shades
        colors = sns.color_palette(pal, len(counts) + 2)[2:][::-1]

        sns.countplot(data=ling_data_creatures, y=col, order=counts.index,
                      hue=col, hue_order=counts.index, palette=colors,
                      legend=False, ax=ax)

        pct = (counts / counts.sum() * 100).round(1)
        for container, p in zip(ax.containers, pct):
            ax.bar_label(container, labels=[f"{p}%"], padding=3, fontsize=14)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Number of respondents")
        ax.set_ylabel("")
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle("Responses to Creature Questions", fontsize=20)
    plt.tight_layout()
    _finish(fig, "eda_creature_bars")


# =============================================================================
# 4. DIMENSION REDUCTION
# =============================================================================

def one_hot_encode(ling_data_final):
    """One 0/1 column per answer (e.g. 'Q065_2.0'). Non-answers (0) get no column.
    Recreated here because ling_location was not cleaned. (Old cell_35)
    """
    qu_columns = ling_data_final.filter(regex=r'^Q\d+$').columns
    answers_zero_as_na = ling_data_final[qu_columns].mask(lambda d: d.eq(0))
    onehot = pd.get_dummies(answers_zero_as_na.astype('category'), prefix_sep='_', dtype=int)

    print("Questions:", len(qu_columns),
          "| Respondents:", onehot.shape[0],
          "| Answer columns:", onehot.shape[1])
    return onehot


def run_pca(onehot):
    """Mean-center (no scaling) and fit PCA. (Compute part of old cell_36)
    Returns pca, pc_scores (all components), explained (%), cumulative (%).
    """
    X_centered = onehot - onehot.mean(axis=0)
    pca = PCA()
    pc_scores = pca.fit_transform(X_centered)
    explained = pca.explained_variance_ratio_ * 100
    cumulative = explained.cumsum()
    return pca, pc_scores, explained, cumulative


def plot_scree(explained, cumulative, n_show=50):
    """Scree plot of the first n_show PCs + variance thresholds. (Plot part of old cell_36)"""
    components = np.arange(1, len(explained) + 1)

    fig, ax = _subplots(figsize=(10, 5))
    ax.plot(components[:n_show], explained[:n_show], color='steelblue',
            marker='o', markersize=3, linewidth=1.5)
    ax.set_xlim(0, n_show + 1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"Variance explained by each component (first {n_show})", fontsize=18)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _finish(fig, "pca_scree")

    for t in [50, 70, 80, 90]:
        print(f"{int((cumulative >= t).argmax() + 1)} components explain at least {t}% of the variance")


def get_pc_scores(pc_scores, onehot, cumulative, n_pcs=10):
    """Keep the top n_pcs (10, from the scree elbow) as a DataFrame. (Old cell_37)"""
    scores = pd.DataFrame(pc_scores[:, :n_pcs], index=onehot.index,
                          columns=[f"PC{i+1}" for i in range(n_pcs)])
    print(f"First {n_pcs} components explain {cumulative[n_pcs-1]:.1f}% of the variance")
    return scores


def plot_pc1_map(scores, states_m, pts_m, boxes_m):
    """Map of respondents colored by PC1 score. (Plot part of old cell_37)"""
    pts = pts_m.copy()
    pts['PC1'] = scores['PC1'].reindex(pts.index)

    fig, ax = _subplots(figsize=(14, 8))
    _draw_base_map(ax, states_m, boxes_m, state_lw=1, box_lw=0.8)
    pts.dropna(subset=['PC1']).plot(ax=ax, column='PC1', cmap='Spectral_r',
                                    markersize=10, edgecolor='grey', linewidth=0.1,
                                    legend=True, zorder=3,
                                    legend_kwds={'shrink': 0.6, 'label': 'PC1 score'})
    ax.set_title("Respondents Colored by PC1 Score (Northeast Dialect)", fontsize=20)
    ax.set_axis_off()
    plt.tight_layout()
    _finish(fig, "pca_pc1_map")


def get_loadings(pca, onehot, n_pcs=10):
    """Loadings of each answer column on the top n_pcs components."""
    return pd.DataFrame(pca.components_[:n_pcs, :].T, index=onehot.columns,
                        columns=[f"PC{i+1}" for i in range(n_pcs)])


def top_answers(loadings, pc='PC1', n=10, show=True):
    """Answers with the largest positive loading on `pc`. (Old cell_38)

    For PC1 these are:
      Q73 gym shoes term, Q105 carbonated beverage, Q80 rain with sun,
      Q56 pantyhose suntan, Q78 used paper, Q95 "the City",
      Q103 water fountain at school, Q86 use of 'cruller',
      Q119 food you take from a restaurant, Q99 road parallel to highway
    """
    top = loadings[pc].nlargest(n)
    if show:
        print(f"=== TOP {n} ANSWERS DRIVING {pc} ===")
        print(top.to_string())
    return top.index


def make_answer_text(question):
    """Return a function that turns a one-hot column like 'Q073_1.0' into
    readable text like 'Q73: sneakers', using the question data.
    (This was needed by the top-PC1 cluster grid but never defined in the script.)
    """
    def answer_text(col):
        q_part, choice = col.split('_')
        qnum = int(q_part[1:])
        choice = int(float(choice))
        letter = chr(ord('a') + choice - 1)
        key = f'ans.{qnum}'
        if key in question:
            ans = question[key]
            row = ans[ans['ans.let'].astype(str).str.strip() == letter]
            if len(row):
                return f"Q{qnum}: {str(row['ans'].iloc[0]).strip()}"
        return f"Q{qnum}: answer {letter}"
    return answer_text


# US Census Bureau's four regions:
# https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf
CENSUS_REGION = {
    **dict.fromkeys(['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'], 'Northeast'),
    **dict.fromkeys(['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'], 'Midwest'),
    **dict.fromkeys(['DE', 'DC', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS',
                     'TN', 'AR', 'LA', 'OK', 'TX'], 'South'),
    **dict.fromkeys(['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI',
                     'OR', 'WA'], 'West'),
}


def run_tsne(pc_scores, onehot, n_pcs=10, perplexity=20, random_state=42):
    """t-SNE on the top n_pcs PCs. SLOW on all respondents. (Compute part of old cell_41)
    Returns a DataFrame with TSNE1, TSNE2 indexed like onehot.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    tsne_results = tsne.fit_transform(pc_scores[:, :n_pcs])
    return pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'], index=onehot.index)


def plot_tsne_map(embedding, states_m, pts_m, boxes_m, perplexity=20):
    """Respondents at their real location, colored by t-SNE dimension 1.
    (Plot part of old cell_41)
    """
    pts = pts_m.loc[pts_m.index.intersection(embedding.index)].copy()
    pts['TSNE1'] = embedding.loc[pts.index, 'TSNE1']

    fig, ax = _subplots(figsize=(14, 8))
    _draw_base_map(ax, states_m, boxes_m)
    pts.plot(ax=ax, column='TSNE1', cmap='Spectral_r',
             markersize=10, edgecolor='grey', linewidth=0.1,
             legend=True, zorder=3,
             legend_kwds={'shrink': 0.6, 'label': 't-SNE Dimension 1'})
    ax.set_title(f"Respondents Colored by t-SNE Dimension 1 (Northeast Dialect) "
                 f"(Perplexity = {perplexity})", fontsize=16)
    ax.set_axis_off()
    plt.tight_layout()
    _finish(fig, "tsne_map")


# =============================================================================
# 5. CLUSTERING
# =============================================================================

def choose_k_kmeans(scores, k_values=range(2, 11), sil_k=range(3, 9), sil_n=5_000):
    """Elbow inertias and silhouette scores for k-means. (Compute part of old cell_44)
    Returns (inertias Series, silhouettes Series).
    """
    inertias = pd.Series(
        {k: KMeans(n_clusters=k, n_init=10, random_state=42).fit(scores).inertia_
         for k in k_values}, name='inertia')

    sample_idx = scores.sample(sil_n, random_state=42).index
    sil = {}
    for k in sil_k:
        labels = pd.Series(KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scores),
                           index=scores.index)
        sil[k] = silhouette_score(scores.loc[sample_idx], labels.loc[sample_idx])
    silhouettes = pd.Series(sil, name='silhouette')
    return inertias, silhouettes


def plot_kmeans_elbow(inertias, silhouettes=None):
    """Elbow plot (+ displays silhouette scores as a table). (Plot part of old cell_44)"""
    fig, ax = _subplots(figsize=(9, 5))
    ax.plot(inertias.index, inertias.values, marker='o', color='steelblue')
    ax.set_xticks(list(inertias.index))
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster sum of squares (inertia)")
    ax.set_title("Choosing k for k-means", fontsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _finish(fig, "kmeans_elbow")

    if silhouettes is not None:
        # Convert dictionary to a pandas DataFrame for tabular display
        sil_df = pd.DataFrame(
            list(silhouettes.items()), 
            columns=['k', 'Silhouette Score']
        )
        
        # Display as a styled table formatted to 3 decimal places without the index
        display(sil_df.style.format({'Silhouette Score': '{:.3f}'}).hide(axis='index'))


def run_kmeans(scores, k=3):
    """Fit k-means on the PC scores. (Compute part of old cell_45)
    Returns (clusters Series, cluster_colors dict). Colors are ordered by
    each cluster's mean PC1 (low -> high = blue -> orange -> red).
    """
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = pd.Series(kmeans.fit_predict(scores), index=scores.index, name='cluster')
    pc1_order = scores['PC1'].groupby(clusters).mean().sort_values().index
    return clusters, _three_colors(pc1_order)


def plot_kmeans_map(clusters, cluster_colors, scores, states_m, pts_m, boxes_m):
    """Map of k-means clusters; largest drawn first so small ones sit on top.
    (Plot part of old cell_45)
    """
    k = clusters.nunique()
    pts_all = pts_m.copy()
    pts_all['cluster'] = clusters.reindex(pts_all.index)

    fig, ax = _subplots(figsize=(14, 8))
    _draw_base_map(ax, states_m, boxes_m)

    sizes = pts_all['cluster'].value_counts()
    for layer, c in enumerate(sizes.index):
        pts = pts_all[pts_all['cluster'] == c]
        pts.plot(ax=ax, color=cluster_colors[c], markersize=6,
                 edgecolor='grey', linewidth=0.1, zorder=3 + layer,
                 label=f"Cluster {int(c) + 1} ({len(pts):,})")

    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), markerscale=3, frameon=False)
    ax.set_title(f"K-Means Clusters (k = {k}) of Respondents' Answers", fontsize=20)
    ax.set_axis_off()
    plt.tight_layout()
    _finish(fig, "kmeans_map")



def plot_top_answers_by_cluster(top_pc1, onehot, clusters, cluster_colors,
                                states_m, pts_m, boxes_m, answer_text, n_cols=5):
    """Small map per top-PC1 answer: who gave it, colored by k-means cluster.
    (Old cell_46)
    """
    pts_all = pts_m.copy()
    pts_all['cluster'] = clusters.reindex(pts_all.index)

    n_rows = math.ceil(len(top_pc1) / n_cols)
    fig, axes = _subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    cluster_order = pts_all['cluster'].value_counts().index  # largest first

    for ax, col in zip(axes, top_pc1):
        chose = onehot.index[onehot[col] == 1]
        pts = pts_all.loc[pts_all.index.intersection(chose)]

        _draw_base_map(ax, states_m, boxes_m, state_lw=0.3, box_lw=0.5)
        for layer, c in enumerate(cluster_order):
            pts[pts['cluster'] == c].plot(ax=ax, color=cluster_colors[c], markersize=1,
                                          zorder=3 + layer)

        ax.set_title(f"{textwrap.fill(answer_text(col), 32)}\n{len(pts):,} respondents", fontsize=11)
        ax.set_axis_off()

    for ax in axes[len(top_pc1):]:
        ax.set_visible(False)

    # Same 1-based cluster numbers as the k-means map
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=8,
                      color=cluster_colors[c], label=f"Cluster {int(c) + 1}")
               for c in sorted(cluster_colors)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=16)

    fig.suptitle("Clusters for Top PCA Questions", fontsize=20)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    _finish(fig, "kmeans_top_pc1_answers")


def run_hierarchical(scores, n_sample=4600, random_state=42):
    """Ward linkage on a ~10% sample of the PC scores. (Compute part of old cell_48)
    Returns (hc_sample, Z).
    """
    hc_sample = scores.sample(n_sample, random_state=random_state)
    Z = linkage(hc_sample, method='ward')
    return hc_sample, Z


def plot_dendrogram(Z, n_sample=4600, p=30):
    """Top of the dendrogram (last p merges). (Plot part of old cell_48)"""
    fig, ax = _subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode='lastp', p=p, show_leaf_counts=True,
               color_threshold=None, ax=ax)
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward Linkage, {n_sample:,} respondents)",
                 fontsize=20)
    ax.set_xlabel("Clusters (number of respondents in parentheses)", fontsize=16)
    ax.set_ylabel("Merge distance", fontsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _finish(fig, "hc_dendrogram")


def cut_hierarchical(Z, hc_sample, scores, k=3):
    """Cut the tree into k clusters. (Compute part of old cell_49)
    Returns (hc_labels Series, hc_colors dict) with colors matching k-means.
    """
    hc_labels = pd.Series(fcluster(Z, t=k, criterion='maxclust'), index=hc_sample.index)
    pc1_order = scores.loc[hc_sample.index, 'PC1'].groupby(hc_labels).mean().sort_values().index
    return hc_labels, _three_colors(pc1_order)


def plot_hierarchical_map(hc_labels, hc_colors, states_m, pts_m, boxes_m):
    """Map of the hierarchical clusters. (Plot part of old cell_49)"""
    k = hc_labels.nunique()
    hc_pts = pts_m.loc[pts_m.index.intersection(hc_labels.index)].copy()
    hc_pts['hc_cluster'] = hc_labels.loc[hc_pts.index]

    fig, ax = _subplots(figsize=(14, 8))
    _draw_base_map(ax, states_m, boxes_m)

    sizes = hc_pts['hc_cluster'].value_counts()
    for layer, c in enumerate(sizes.index):
        pts = hc_pts[hc_pts['hc_cluster'] == c]
        pts.plot(ax=ax, color=hc_colors[c], markersize=10,
                 edgecolor='grey', linewidth=0.1, zorder=3 + layer,
                 label=f"Cluster {c} ({len(pts):,})")

    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), markerscale=2, frameon=False)
    ax.set_title(f"Hierarchical clusters (Ward, k = {k}) of {len(hc_labels):,} respondents",
                 fontsize=20)
    ax.set_axis_off()
    plt.tight_layout()
    _finish(fig, "hc_map")


def compare_clusterings(clusters, hc_labels):
    """ARI between k-means and hierarchical on the same respondents + row-% crosstab.
    (Old cell_50) Returns (ari, ct_pct).
    """
    km_sample = clusters.loc[hc_labels.index]
    ari = adjusted_rand_score(km_sample, hc_labels)
    ct_pct = pd.crosstab(hc_labels, km_sample + 1, rownames=['Hierarchical'],
                         colnames=['k-means'], normalize='index').mul(100).round(1)
    return ari, ct_pct


# =============================================================================
# 6. STABILITY
# =============================================================================

def _ari_summary(ari, label):
    return pd.DataFrame({'Runs': [len(ari)], 'Mean ARI': [ari.mean()],
                         'Min ARI': [ari.min()], 'Max ARI': [ari.max()]}, index=[label])


ARI_FORMAT = {'Mean ARI': '{:.3f}', 'Min ARI': '{:.3f}', 'Max ARI': '{:.3f}'}


def kmeans_seed_stability(clusters, scores, k=3, n_seeds=20):
    """Re-run k-means from n_seeds random starts (n_init=1 each); ARI vs original.
    (Compute part of old cell_53) Returns (seed_ari Series, summary table).
    """
    reference = clusters.loc[scores.index]
    seed_ari = pd.Series(
        [adjusted_rand_score(reference,
                             KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(scores))
         for s in range(n_seeds)],
        name='ARI')
    return seed_ari, _ari_summary(seed_ari, 'K-means: different starting points')


def plot_seed_stability(seed_ari, k=3):
    """Histogram of k-means ARI across starting points. (Plot part of old cell_53)"""
    fig, ax = _subplots(figsize=(9, 5))
    ax.hist(seed_ari, bins=15, color=plt.get_cmap('Spectral_r')(0.1),
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Adjusted Rand Index vs. Original K-Means Clustering", fontsize=16)
    ax.set_ylabel("Number of Runs", fontsize=16)
    ax.set_title(f"K-Means Stability Across {len(seed_ari)} Random Starting Points (k = {k})",
                 fontsize=20)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _finish(fig, "stability_kmeans_seeds")


def hierarchical_stability(scores, hc_sample, hc_labels, k=3, n_runs=20):
    """New random sample of the same size each run, Ward-clustered; each original
    sampled respondent is assigned to the nearest new cluster center; ARI vs
    original. (Old cell_54) Returns (hc_ari Series, summary table).
    """
    hc_ari = []
    for s in range(n_runs):
        samp = scores.sample(len(hc_sample), random_state=s)
        labels = fcluster(linkage(samp, method='ward'), t=k, criterion='maxclust')
        centers = samp.groupby(labels).mean()
        assigned = pairwise_distances_argmin(hc_sample, centers)
        hc_ari.append(adjusted_rand_score(hc_labels, assigned))
    hc_ari = pd.Series(hc_ari, name='ARI')
    return hc_ari, _ari_summary(hc_ari, 'Hierarchical: different random samples')


def stability_table(*summaries):
    """Stack ARI summary tables and format them for display."""
    return pd.concat(summaries).style.format(ARI_FORMAT)


# =============================================================================
# Run everything as a script:  python lab2_script.py
# =============================================================================

def main():
    set_plot_style()
    states, ling_data, ling_location, question = load_data()
    data_overview(states, ling_data, ling_location)

    ling_data_final, states_simpl, summary_df = clean_data(ling_data, states, verbose=True)
    display(summary_df)

    creatures = build_creatures_df(ling_data_final)
    show_response_tables(creatures)
    states_m, pts_m, boxes_m = project_for_maps(creatures, states, states_simpl)
    map_q65_grid(states_m, pts_m, boxes_m)
    map_q66_grid(states_m, pts_m, boxes_m)
    map_q65(states_m, pts_m, boxes_m)
    map_q66(states_m, pts_m, boxes_m)
    both, ct_pct = crosstab_q66_q65(creatures)
    display(ct_pct)
    display(test_q66_predicts_q65(both))
    plot_creature_bars(creatures)

    onehot = one_hot_encode(ling_data_final)
    pca, pc_scores, explained, cumulative = run_pca(onehot)
    plot_scree(explained, cumulative)
    scores = get_pc_scores(pc_scores, onehot, cumulative)
    plot_pc1_map(scores, states_m, pts_m, boxes_m)
    loadings = get_loadings(pca, onehot)
    top_pc1 = top_answers(loadings, 'PC1')
    embedding = run_tsne(pc_scores, onehot)
    plot_tsne_map(embedding, states_m, pts_m, boxes_m)

    inertias, silhouettes = choose_k_kmeans(scores)
    plot_kmeans_elbow(inertias, silhouettes)
    clusters, cluster_colors = run_kmeans(scores, k=3)
    plot_kmeans_map(clusters, cluster_colors, scores, states_m, pts_m, boxes_m)
    plot_top_answers_by_cluster(top_pc1, onehot, clusters, cluster_colors,
                                states_m, pts_m, boxes_m, make_answer_text(question))
    hc_sample, Z = run_hierarchical(scores)
    plot_dendrogram(Z)
    hc_labels, hc_colors = cut_hierarchical(Z, hc_sample, scores, k=3)
    plot_hierarchical_map(hc_labels, hc_colors, states_m, pts_m, boxes_m)
    ari, ct = compare_clusterings(clusters, hc_labels)
    print(f"Adjusted Rand index (k-means vs. hierarchical): {ari:.3f}")
    display(ct)

    seed_ari, seed_summary = kmeans_seed_stability(clusters, scores)
    plot_seed_stability(seed_ari)
    hc_ari, hc_summary = hierarchical_stability(scores, hc_sample, hc_labels)
    display(stability_table(seed_summary, hc_summary))


if __name__ == "__main__":
    main()