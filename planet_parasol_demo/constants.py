from pathlib import Path


SIM_START_YEAR = 1950
IGNORE_DIR = SIM_START_YEAR == 1750
SIM_END_YEAR = 2100
CANVAS_START_YEAR = 2024
CANVAS_END_YEAR = 2100
PLOT_START_YEAR = 1950
PLOT_END_YEAR = 2100
NUM_MEMBERS = 100
NUM_EMULATORS = 100
MAX_CO2 = 155
MAX_SAI = 31
REGIONAL_MODEL_NAMES = [
    "CESM2-WACCM",
    "CNRM-ESM2-1",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "UKESM1-0-LL"
]
# CESM2-WACCM dimensions
NUM_LAT = 192
NUM_LON = 288

DATA_DIR = Path("data")
CO2_IMG_PATH = DATA_DIR / "co2" / "co2_img.png"
CO2_NP_PATH = DATA_DIR / "co2" / "co2_input.npy"
SAI_IMG_PATH = DATA_DIR / "sai" / "sai_img.png"
SAI_NP_PATH = DATA_DIR / "sai" / "sai_input.npy"
REGIONAL_MODEL_DIR = Path("regional_data/models_arise-sai-1.0_arise-sai-1.5-2045")
REGIONAL_DATA_DIR = Path("regional_data/data")

F_INIT_DIR = DATA_DIR / "f_no_sai"
TEMP_DATA_DIR = DATA_DIR / "temp"

# Sorted by end of century temp
SCENARIOS = ["ssp119", "ssp126", "ssp534-over", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
BASE_SCENARIO = "ssp245" # Base scenario for the draw your own option
FANCY_SSP_TITLES = {
    "ssp119": r"SSP1-1.9",
    "ssp126": r"SSP1-2.6",
    "ssp434": r"SSP4-3.4",
    "ssp534-over": r"SSP5-3.4",
    "ssp245": r"SSP2-4.5",
    "ssp460": r"SSP4-6.0",
    "ssp370": r"SSP3-Baseline",
    "ssp585": r"SSP5-Baseline",
}
# Sort FANCY_SSP_TITLES by SCENARIOS
FANCY_SSP_TITLES = {scenario: FANCY_SSP_TITLES[scenario] for scenario in SCENARIOS}
FANCY_SSP_TITLES_SUFFIXES = {
    "SSP1-1.9": "SSP1-1.9: Aggressive climate action.",
    "SSP1-2.6": "SSP1-2.6: Strong climate action.",
    "SSP4-3.4": "SSP4-3.4: High inequality, some climate action.",
    "SSP5-3.4": "SSP5-3.4: High reliance on fossil fuels, high levels of carbon dioxide removal.",
    "SSP2-4.5": "SSP2-4.5: Balanced efforts.",
    "SSP4-6.0": "SSP4-6.0: High inequality, minimal climate action.",
    "SSP3-Baseline": "SSP3-Baseline (SSP3-7.0): Geopolitical conflict, minimal climate action.",
    "SSP5-Baseline": "SSP5-Baseline (SSP5-8.5): High fossil fuel use.",
    "Draw your own emissions scenario!": "Draw your own emissions scenario!"
}

DRAW_SSP_SCENARIO = "Draw your own emissions scenario!"
REVERSE_FANCY_SSP_TITLES = {
    title: scenario for scenario, title in FANCY_SSP_TITLES.items()
}
REVERSE_FANCY_SSP_TITLES_SUFFIXES = {
    suffix: title for title, suffix in FANCY_SSP_TITLES_SUFFIXES.items()
}
LEVEL1_SAI = "Light Injection"
LEVEL2_SAI = "Mild Injection"
LEVEL3_SAI = "Moderate Injection"
LEVEL4_SAI = "Considerable Injection"
LEVEL5_SAI = "Substantial Injection"
SAI_TITLES = [LEVEL1_SAI, LEVEL2_SAI, LEVEL3_SAI, LEVEL4_SAI, LEVEL5_SAI]
SAI_VALUES = [5, 10, 15, 20, 25] 
DRAW_SAI_SCENARIO = "Draw your own SAI scenario!"

AR6_COLORS = {
    "ssp119": "#4a90e2", # Blue
    "ssp126": "#8b4513", # Brown
    "ssp245": "#2ecc71", # Green
    "ssp370": "#f39c12", # Orange
    "ssp434": "#3498db", # Blue
    "ssp460": "#ffd700", # Yellow
    "ssp534-over": "#27ae60", # Green
    "ssp585": "#9b59b6", # Purple
}

TEMP_COLORS = {
    (0, 2): "#FFD700", # Yellow
    (2, 4): "#FF8C00", # Gold
    (4, 6): "#FF4500", # Dark Orange
    (6, 8): "#8B0000", # Red Orange
}

IMPACTS = {
    2: [
        "Increased heatwaves and extreme weather events.",
        "Rising sea levels threatening coastal communities."
    ],
    4: [
        "Catastrophic sea-level rise endangering coastal areas.",
        "Loss of biodiversity and collapse of ecosystems."
    ],
    6: [
        "Drastic loss of habitable land and mass displacement.",
        "Global food systems collapse leading to famine."
    ],
    8: [
        "Collapse of global food systems, widespread famine.",
        "Severe disruptions to ecosystems and biodiversity."
    ],
}

BASE_FIGSIZE = (12, 4)
BASE_CANVAS_SIZE = (BASE_FIGSIZE[0] * 100, BASE_FIGSIZE[1] * 100)
BASE_FONTSIZE = 18
BASE_DPI = 50
