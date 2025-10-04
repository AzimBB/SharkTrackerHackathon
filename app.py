import os
import numpy as np
import pandas as pd
import xarray as xr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "nasa"

# Consolidated list of all shark models with default preferences and weights
# Weights MUST sum to 1.0
SHARK_MODELS = {
    # Default is the original Great White Shark model
    "Great White Shark": {
        "weights": {"SST": 0.35, "CHL-A": 0.10, "SSHa": 0.35, "BATHY": 0.20},
        "preferences": {
            "SST": {"optimal": (15, 22), "tolerance": (10, 28), "units": "°C"},
            "CHL-A": {"optimal": (0.2, 2.0), "tolerance": (0.1, 5.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.05, 0.2), "tolerance": (-0.1, 0.3), "units": "m"},
            "BATHY": {"optimal": (20, 200), "tolerance": (5, 1000), "units": "m"},
        },
    },
    "Tiger Shark": {
        "weights": {"SST": 0.40, "CHL-A": 0.10, "SSHa": 0.20, "BATHY": 0.30},
        "preferences": {
            "SST": {"optimal": (22, 28), "tolerance": (20, 30), "units": "°C"},
            "CHL-A": {"optimal": (0.2, 2.0), "tolerance": (0.1, 5.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.05, 0.2), "tolerance": (-0.1, 0.3), "units": "m"},
            "BATHY": {"optimal": (5, 100), "tolerance": (1, 500), "units": "m"},
        },
    },
    "Whale Shark": {
        "weights": {"SST": 0.30, "CHL-A": 0.40, "SSHa": 0.10, "BATHY": 0.20},
        "preferences": {
            "SST": {"optimal": (26, 30), "tolerance": (21, 32), "units": "°C"},
            "CHL-A": {"optimal": (1.0, 5.0), "tolerance": (0.5, 10.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.05, 0.2), "tolerance": (-0.1, 0.3), "units": "m"},
            "BATHY": {"optimal": (20, 500), "tolerance": (5, 2000), "units": "m"},
        },
    },
    "Bull Shark": {
        # Note: Salinity is a key factor not included in your data. CHL-A/BATHY are proxies for coastal habitat.
        "weights": {"SST": 0.40, "CHL-A": 0.20, "SSHa": 0.10, "BATHY": 0.30},
        "preferences": {
            "SST": {"optimal": (20, 28), "tolerance": (18, 30), "units": "°C"},
            "CHL-A": {"optimal": (0.5, 3.0), "tolerance": (0.1, 6.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.05, 0.2), "tolerance": (-0.1, 0.3), "units": "m"},
            "BATHY": {"optimal": (1, 50), "tolerance": (0, 200), "units": "m"},
        },
    },
    "Greenland Shark": {
        "weights": {"SST": 0.20, "CHL-A": 0.05, "SSHa": 0.05, "BATHY": 0.70},
        # **SST logic is reversed in normalize_preference for this species**
        "preferences": {
            "SST": {"optimal": (-1.8, 5.0), "tolerance": (-2.0, 10.0), "units": "°C", "is_low_temp_opt": True},
            "CHL-A": {"optimal": (0.1, 1.0), "tolerance": (0.05, 3.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.0, 0.0), "tolerance": (-0.1, 0.1), "units": "m"}, # Very low SSHA preference
            "BATHY": {"optimal": (400, 1500), "tolerance": (200, 3000), "units": "m"},
        },
    },
    "Reef Shark": {
        "weights": {"SST": 0.40, "CHL-A": 0.05, "SSHa": 0.05, "BATHY": 0.50},
        "preferences": {
            "SST": {"optimal": (24, 28), "tolerance": (22, 30), "units": "°C"},
            "CHL-A": {"optimal": (0.1, 1.0), "tolerance": (0.05, 2.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.05, 0.2), "tolerance": (-0.1, 0.3), "units": "m"},
            "BATHY": {"optimal": (10, 60), "tolerance": (5, 200), "units": "m"},
        },
    },
    "Shortfin Mako Shark": {
        "weights": {"SST": 0.45, "CHL-A": 0.05, "SSHa": 0.40, "BATHY": 0.10},
        "preferences": {
            "SST": {"optimal": (20, 28), "tolerance": (18, 30), "units": "°C"},
            "CHL-A": {"optimal": (0.1, 0.5), "tolerance": (0.05, 1.0), "units": "mg/m^3"}, # Prefers low CHLA for clear open ocean
            "SSHa": {"optimal": (0.1, 0.3), "tolerance": (0.0, 0.4), "units": "m"},
            "BATHY": {"optimal": (500, 5000), "tolerance": (150, 5000), "units": "m"}, # Deep-water preference
        },
    },
    "shortfin_mako": {
        "weights": {"SST": 0.40, "CHL-A": 0.10, "SSHa": 0.30, "BATHY": 0.20},
        "preferences": {
            "SST": {"optimal": (16, 25), "tolerance": (12, 28), "units": "°C"},
            "CHL-A": {"optimal": (0.1, 0.8), "tolerance": (0.05, 2.0), "units": "mg/m^3"},
            "SSHa": {"optimal": (0.05, 0.2), "tolerance": (-0.1, 0.3), "units": "m"},
            "BATHY": {"optimal": (100, 1000), "tolerance": (50, 2000), "units": "m"},
        },
    },
}


TIME_START = pd.to_datetime("2025-08-08")
TIME_END = pd.to_datetime("2025-08-28")

# File mapping (kept for context, no change needed)
FILE_MAP = {
    # ... (Your existing FILE_MAP data) ...
    '2025-08-08': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-09': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-10': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-11': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-12': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-13': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-14': ('AQUA_MODIS.20250805_20250812.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250805_20250812.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250808_20250814.nc'),
    '2025-08-15': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-16': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-17': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-18': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-19': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-20': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-21': ('AQUA_MODIS.20250813_20250820.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250813_20250820.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250815_20250821.nc'),
    '2025-08-22': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
    '2025-08-23': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
    '2025-08-24': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
    '2025-08-25': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
    '2025-08-26': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
    '2025-08-27': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
    '2025-08-28': ('AQUA_MODIS.20250821_20250828.L3m.8D.CHL.chlor_a.4km.NRT.nc', 'AQUA_MODIS.20250821_20250828.L3m.8D.SST.sst.4km.nc', 'nrt_global_allsat_phy_l4_20250822_20250828.nc'),
}
# FILE_MAP = {
#     # ... (Your existing FILE_MAP data) ...
#     '2025-08-08': ('AQUA_MODIS.20250808.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250808.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-09': ('AQUA_MODIS.20250809.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250809.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-10': ('AQUA_MODIS.20250810.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250810.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-11': ('AQUA_MODIS.20250811.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250811.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-12': ('AQUA_MODIS.20250812.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250812.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-13': ('AQUA_MODIS.20250813.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250813.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-14': ('AQUA_MODIS.20250814.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250814.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-15': ('AQUA_MODIS.20250815.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250815.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-16': ('AQUA_MODIS.20250816.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250816.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-17': ('AQUA_MODIS.20250817.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250817.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-18': ('AQUA_MODIS.20250818.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250818.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-19': ('AQUA_MODIS.20250819.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250819.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-20': ('AQUA_MODIS.20250820.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250820.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-21': ('AQUA_MODIS.20250821.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250821.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-22': ('AQUA_MODIS.20250822.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250822.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-23': ('AQUA_MODIS.20250823.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250823.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-24': ('AQUA_MODIS.20250824.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250824.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-25': ('AQUA_MODIS.20250825.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250825.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-26': ('AQUA_MODIS.20250826.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250826.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-27': ('AQUA_MODIS.20250827.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250827.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
#     '2025-08-28': ('AQUA_MODIS.20250828.L3m.DAY.CHL.chlor_a.4km.nc', 'AQUA_MODIS.20250828.L3m.DAY.SST.sst.4km.nc', 'cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D_1759545339589.nc'),
# }

BATHY_FILE = os.path.join(DATA_DIR, "GEBCO_2025_sub_ice.nc")

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# UTILS
# -----------------------------
def vectorized_normalize_preference(arr, prefs):
    vals = np.array(arr, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    nan_mask = np.isnan(vals)
    out[nan_mask] = 0.0

    opt_min, opt_max = prefs["optimal"]
    tol_min, tol_max = prefs["tolerance"]
    is_low_temp_opt = prefs.get("is_low_temp_opt", False)

    if is_low_temp_opt:
        vals_proc = -vals
        opt_min_f, opt_max_f = -opt_max, -opt_min
        tol_min_f, tol_max_f = -tol_max, -tol_min

        mask_opt = (vals_proc >= opt_min_f) & (vals_proc <= opt_max_f)
        mask_left = (vals_proc >= tol_min_f) & (vals_proc < opt_min_f)
        mask_right = (vals_proc > opt_max_f) & (vals_proc <= tol_max_f)

        out[mask_opt] = 1.0
        if (opt_min_f - tol_min_f) != 0:
            out[mask_left] = (vals_proc[mask_left] - tol_min_f) / (opt_min_f - tol_min_f)
        if (tol_max_f - opt_max_f) != 0:
            out[mask_right] = (tol_max_f - vals_proc[mask_right]) / (tol_max_f - opt_max_f)
    else:
        mask_opt = (vals >= opt_min) & (vals <= opt_max)
        mask_left = (vals >= tol_min) & (vals < opt_min)
        mask_right = (vals > opt_max) & (vals <= tol_max)

        out[mask_opt] = 1.0
        if (opt_min - tol_min) != 0:
            out[mask_left] = (vals[mask_left] - tol_min) / (opt_min - tol_min)
        if (tol_max - opt_max) != 0:
            out[mask_right] = (tol_max - vals[mask_right]) / (tol_max - opt_max)

    return np.clip(out, 0.0, 1.0)


def vectorized_eddy_score(ssha_arr, eddy_pref):
    ssha = np.array(ssha_arr, dtype=float)
    out = np.full_like(ssha, 0.5, dtype=float)
    pos_mask = ssha > 0.05
    neg_mask = ssha < -0.05
    if eddy_pref.get("preferred", "anticyclonic") == "anticyclonic":
        out[pos_mask] = 1.0
        out[neg_mask] = 0.0
    else:
        out[pos_mask] = 0.0
        out[neg_mask] = 1.0
    out[np.isnan(ssha)] = 0.0
    return out


def season_score_scalar(date, season_pref):
    peak = season_pref.get("peak_month", 1)
    month = pd.to_datetime(date).month
    angle_peak = 2 * np.pi * (peak / 12.0)
    angle_month = 2 * np.pi * (month / 12.0)
    return float((np.cos(angle_peak - angle_month) + 1.0) / 2.0)


def get_dataset_paths(date_str):
    if date_str not in FILE_MAP:
        raise ValueError(f"No mapped files for {date_str}")
    chla_file, sst_file, ssha_file = FILE_MAP[date_str]
    return (
        os.path.join(DATA_DIR, chla_file),
        os.path.join(DATA_DIR, sst_file),
        os.path.join(DATA_DIR, ssha_file),
    )

# -----------------------------
# ENDPOINT
# -----------------------------
@app.route("/calculate_hsi", methods=["POST"])
def calculate_hsi():
    try:
        data = request.json

        lat_min, lat_max = float(data["lat_min"]), float(data["lat_max"])
        lon_min, lon_max = float(data["lon_min"]), float(data["lon_max"])
        target_date_str = data["date"]

        shark_type = data.get("shark_type", "Great White Shark")
        print(shark_type)
        if shark_type not in SHARK_MODELS:
            return jsonify({"error": f"Unknown shark type: {shark_type}"}), 400

        model_data = SHARK_MODELS[shark_type]
        current_weights = model_data["weights"].copy()
        print(current_weights)
        current_preferences = {k: v.copy() for k, v in model_data["preferences"].items()}
        print(current_preferences)

        # override weights if user provides
        user_weights = data.get("weights", {})
        if user_weights:
            total_weight = sum(user_weights.values())
            if not np.isclose(total_weight, 1.0):
                return jsonify({"error": "Weights must sum to 1.0"}), 400
            current_weights.update(user_weights)

        print(current_weights)
        # override prefs if provided
        user_prefs = data.get("preferences", {})
        for var, new_prefs in user_prefs.items():
            if var in current_preferences:
                if "optimal" in new_prefs:
                    current_preferences[var]["optimal"] = tuple(new_prefs["optimal"])
                if "tolerance" in new_prefs:
                    current_preferences[var]["tolerance"] = tuple(new_prefs["tolerance"])

        print(user_prefs)

        target_date = pd.to_datetime(target_date_str)
        if not (TIME_START <= target_date <= TIME_END):
            return jsonify({"error": "Date out of range"}), 400

        chla_path, sst_path, ssha_path = get_dataset_paths(target_date_str)

        N_POINTS = int(data.get("n_points", 100))
        lats = np.linspace(lat_min, lat_max, N_POINTS)
        lons = np.linspace(lon_min, lon_max, N_POINTS)
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)

        # ---------------- DATA EXTRACTION ----------------
        with xr.open_dataset(chla_path) as ds_chla, xr.open_dataset(sst_path) as ds_sst, \
                xr.open_dataset(ssha_path) as ds_ssha_raw, xr.open_dataset(BATHY_FILE) as ds_bathy:

            ds_ssha = ds_ssha_raw.rename({"latitude": "lat", "longitude": "lon"})

            chla_interp = ds_chla["chlor_a"].interp(lat=(("y", "x"), lat_mesh), lon=(("y", "x"), lon_mesh))
            sst_interp = ds_sst["sst"].interp(lat=(("y", "x"), lat_mesh), lon=(("y", "x"), lon_mesh))

            try:
                ssha_interp = ds_ssha["sla"].sel(time=target_date_str).interp(lat=(("y", "x"), lat_mesh), lon=(("y", "x"), lon_mesh))
            except Exception:
                ssha_interp = ds_ssha["sla"].interp(lat=(("y", "x"), lat_mesh), lon=(("y", "x"), lon_mesh))

            depth_interp = ds_bathy["elevation"].interp(lat=(("y", "x"), lat_mesh), lon=(("y", "x"), lon_mesh))

            chla_vals = chla_interp.values
            sst_vals = sst_interp.values
            ssha_vals = ssha_interp.values
            depth_vals = depth_interp.values

        # Convert depth: positive below sea level
        depth_inv = -np.array(depth_vals, dtype=float)
        ocean_mask = depth_inv > 0

        sst_norm = vectorized_normalize_preference(sst_vals, current_preferences["SST"])
        chla_norm = vectorized_normalize_preference(chla_vals, current_preferences["CHL-A"])
        ssha_norm = vectorized_normalize_preference(ssha_vals, current_preferences["SSHa"])
        depth_norm = vectorized_normalize_preference(depth_inv, current_preferences["BATHY"])
        eddy_norm = vectorized_eddy_score(ssha_vals, {"preferred": "anticyclonic"})


        season_scalar = season_score_scalar(target_date, {"peak_month": target_date.month})
        season_norm = np.full_like(sst_norm, season_scalar)

        print("Destianted weights : ",current_weights)
        final_hsi = (
                current_weights.get("SST", 0) * sst_norm +
                current_weights.get("CHL-A", 0) * chla_norm +
                current_weights.get("SSHa", 0) * ssha_norm +
                current_weights.get("BATHY", 0) * depth_norm +
                current_weights.get("EDDY", 0) * eddy_norm +
                current_weights.get("SEASON", 0) * season_norm
        )
        # Remove any singleton dimensions (e.g. time dimension)
        final_hsi = np.squeeze(final_hsi)

        # Now apply the ocean mask
        final_hsi[~ocean_mask] = np.nan



        results = []
        for i in range(final_hsi.shape[0]):
            for j in range(final_hsi.shape[1]):
                if not np.isnan(final_hsi[i, j]):
                    results.append({
                        "lat": float(lat_mesh[i, j]),
                        "lon": float(lon_mesh[i, j]),
                        "hsi": float(final_hsi[i, j])
                    })

        return jsonify({"data": results, "message": f"HSI computed for {len(results)} points"})

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500

@app.route('/',  methods=["GET"])
def index():
    return render_template('index.html')

# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)