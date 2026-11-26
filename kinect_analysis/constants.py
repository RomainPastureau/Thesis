import os.path as op

# General
SYSTEM = "Kinect"
KINEMATIC_MEASURES = ["velocity", "acceleration", "jerk", "snap"]
DERIVATIVES = ["envelope", "intensity", "pitch", "f1", "f2"]
REREF_SUFFIX = "_rSpineBase"

# Paths
PATH_PROJECT = op.join("Project/")
PATH_DATA = op.join(PATH_PROJECT, "Data", "01 Kinect")

PATH_MOCAP = op.join(PATH_DATA, "Mocap")
PATH_MOCAP_SINGLE_FILES = op.join(PATH_MOCAP, "01_Original_gesture")
PATH_MOCAP_ORIGINAL = op.join(PATH_MOCAP, "01 Merged mocap")
PATH_MOCAP_PREPROCESSED = op.join(PATH_MOCAP, "02 Preprocessed mocap")

PATH_AUDIO = op.join(PATH_DATA, "Audio")
PATH_AUDIO_PREPROCESSED = op.join(PATH_AUDIO, "02 Preprocessed audio")
PATH_AUDIO_DERIVATIVE = op.join(PATH_AUDIO, "03 Audio derivative")

PATH_VIDEO = op.join(PATH_DATA, "Audiovisual", "Bodies")

PATH_INFO = op.join(PATH_PROJECT, "Documents", "Results", "Recordings", "Recordings Kinect.xlsx")

PATH_DATAFRAME = op.join(PATH_DATA, "Dataframes")
PATH_RESULTS = op.join(PATH_DATA, "Results")
PATH_PLOTS = op.join(PATH_DATA, "Plots")

# Metadata
METADATA_KEYS = {"SensorID", "ImageWidth", "ImageHeight", "DepthWidth", "DepthHeight", "SensorIntrinsics"}

# Subjects
ALL_SUBJECTS = ["K01", "K02", "K03", "K04", "K05", "K06"]
SUBJECTS_NO_OUTLIERS = ["K01", "K02", "K03", "K04", "K05"]

# Exclusion
EXCLUDED_SUBJECTS = []
EXCLUDED_RECORDINGS = {"K02": ["R001", "R002", "R003"], "K03": ["R003"]}
                       # "K05": ["R001", "R002", "R003"]}
# EXCLUDED_RECORDINGS = {}

# Frequency
FS = 20.0
CUTOFF_FREQ_LOW = 0.2
CUTOFF_FREQ_HIGH = 4

# Display
VIDEO_SHIFTS = {"K01": (-331, -179), "K02": (40, 180), "K03": (75, 150), "K04": (80, 70),
                "K05": (-385, -224), "K06": (100, 140)}

# Plot colors
SUBJECT_COLORS = {"K01": "#6E3F7C", "K02": "#028090", "K03": "#758E4F", "K04": "#F6AE2D",
    "K05": "#E67425", "K06": "#B82F2B", "S1_pure_0.5Hz": "#6E3F7C", "S2_pure_1.5Hz": "#028090",
    "S3_dual_peak": "#758E4F", "S4_white_noise": "#F6AE2D", "S5_pink_noise": "#E67425", }

SHOW_COLORS = {"Wile E. Coyote": "#FFAB66", "Pink Panther": "#FF66FF", "Tom & Jerry": "#7EC6E7", }

CONDITION_COLORS = {"Unmasked": "#F6AE2D", "Masked": "#028090"}

PER_JOINT_GREYS = {"Head": "0.75",  # light grey
    "HandLeft": "0.55",  # medium grey
    "HandRight": "0.35",  # dark grey
    "Average of all joints": "0.00",  # black
}

# Joints
UPPER_BODY_JOINTS = {"Head", "Neck", "SpineShoulder", "SpineMid", "SpineBase", "ShoulderLeft", "ElbowLeft", "WristLeft",
                     "HandLeft", "ShoulderRight", "ElbowRight", "WristRight", "HandRight", "HipLeft", "HipRight"}
UPPER_BODY_JOINTS_REREF = {"Head", "Neck", "SpineShoulder", "SpineMid", "ShoulderLeft", "ElbowLeft", "WristLeft",
                           "HandLeft", "ShoulderRight", "ElbowRight", "WristRight", "HandRight", "HipLeft", "HipRight"}
LOWER_BODY_JOINTS = {"KneeLeft", "KneeRight", "AnkleLeft", "AnkleRight", "FootLeft", "FootRight"}
ALL_JOINTS = UPPER_BODY_JOINTS | LOWER_BODY_JOINTS
BILATERAL_PAIRS = [("ShoulderRight", "ShoulderLeft"), ("ElbowRight", "ElbowLeft"), ("WristRight", "WristLeft"),
                   ("HandRight", "HandLeft")]
BILATERAL_PAIRS_UPPER = [("ShoulderRight", "ShoulderLeft"), ("ElbowRight", "ElbowLeft"), ("WristRight", "WristLeft"),
                         ("HandRight", "HandLeft")]
JOINT_GROUPS = {"Axial": ["Head", "Neck", "SpineShoulder", "SpineMid"],
    "Upper limb": ["ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft", "ShoulderRight", "ElbowRight", "WristRight",
                   "HandRight"],
    "Lower limb": ["HipLeft", "KneeLeft", "AnkleLeft", "FootLeft", "HipRight", "KneeRight", "AnkleRight",
                   "FootRight"]}

# Matplotlib-specific
from matplotlib import font_manager
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="whitegrid", font_scale=1.2, rc=custom_params)
SAVE_KW = dict(dpi=150, bbox_inches="tight")

PATH_METROPOLIS = r"Metropolis.otf"
PATH_MERRIWEATHER = r"Merriweather.otf"

for path in Path(PATH_METROPOLIS).glob("*.otf"):
    font_manager.fontManager.addfont(str(path))
for path in Path(PATH_MERRIWEATHER).glob("*.otf"):
    font_manager.fontManager.addfont(str(path))

plt.rcParams["font.family"] = "Metropolis"
