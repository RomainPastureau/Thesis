import os.path as op

# General
SYSTEM = "Qualisys"
KINEMATIC_MEASURES = ["velocity", "acceleration", "jerk", "snap"]
DERIVATIVES = ["envelope", "intensity", "pitch", "f1", "f2"]
REREF_SUFFIX = "_rWaistBack"

# Paths
PATH_PROJECT = op.join("Project")
PATH_DATA = op.join(PATH_PROJECT, "Data", "02 Qualisys")

PATH_MOCAP = op.join(PATH_DATA, "Mocap")
PATH_MOCAP_ORIGINAL = op.join(PATH_MOCAP, "01 Original mocap")
PATH_MOCAP_PREPROCESSED = op.join(PATH_MOCAP, "02 Preprocessed mocap")

PATH_AUDIO = op.join(PATH_DATA, "Audio")
PATH_AUDIO_CLEAN = op.join(PATH_AUDIO, "01 Clean audio")
PATH_AUDIO_PREPROCESSED = op.join(PATH_AUDIO, "02 Preprocessed audio")
PATH_AUDIO_DERIVATIVE = op.join(PATH_AUDIO, "03 Audio derivative")

PATH_MIQUS = op.join("Qualisys", "Mocap")
PATH_VIDEO = op.join(PATH_DATA, "Audiovisual")
PATH_VIDEO_FRONT = op.join(PATH_VIDEO, "Front")
PATH_VIDEO_SIDE = op.join(PATH_VIDEO, "Side")

PATH_INFO = op.join(PATH_PROJECT, "Documents", "Results", "Recordings", "Recordings Qualisys.xlsx")

PATH_DATAFRAME = op.join(PATH_DATA, "Dataframes")
PATH_RESULTS = op.join(PATH_DATA, "Results")
PATH_PLOTS = op.join(PATH_DATA, "Plots")

# Subjects
ALL_SUBJECTS = ["S" + str(i).zfill(3) for i in range(1, 25)]
ALL_SUBJECTS_MINUS_S004 = ["S" + str(i).zfill(3) for i in range(1, 4)] + ["S" + str(i).zfill(3) for i in range(5, 25)]
SUBJECTS_NO_OUTLIERS = ["S001", "S002", "S003", "S005", "S006", "S007", "S008", "S009",
                        "S010", "S011", "S012", "S014", "S015", "S016", "S017",
                        "S018", "S019", "S020", "S021", "S022", "S024"]
SUBJECT_ANON = {"S" + str(i).zfill(3): "Q" + str(i).zfill(2) for i in range(1, 25)}

# Exclusion
EXCLUDED_SUBJECTS = ["S004"]
EXCLUDED_RECORDINGS = {"S001": ["R001"],
                       "S002": ["R001"],
                       "S007": ["R004"],
                       "S008": ["R001", "R002", "R003", "R004", "R005", "R006"],
                       "S009": ["R001"],
                       "S020": ["R006"],
                       "S021": ["R001", "R002", "R003"],
                       "S023": ["R018", "R019", "R020", "R021", "R022", "R023", "R024", "R025", "R026", "R027",
                                "R028", "R029", "R030"]}

# Frequency
FS = 50
CUTOFF_FREQ_LOW = 0.2
CUTOFF_FREQ_HIGH = 20

# Plot colors
SUBJECT_COLORS = {"S001": "#6E3F7C",  # dark plum
    "S002": "#028090",  # dirty blue
    "S003": "#758E4F",  # dark olive
    "S005": "#F6AE2D",  # sunflower
    "S006": "#E67425",  # apricot
    "S007": "#B82F2B",  # cherry
    "S008": "#AF7DBD",  # light plum
    "S009": "#27E1FC",  # sky
    "S010": "#ADC08F",  # pastel green
    "S011": "#FACF81",  # pastel yellow
    "S012": "#F0AA7B",  # light apricot
    "S013": "#DF7875",  # strawberry
    "S014": "#D26ED4",  # mauve
    "S015": "#3333CC",  # navy
    "S016": "#5AAB4F",  # shrub
    "S017": "#145A32",  # dark teal
    "S018": "#C08A54",  # brown
    "S019": "#666699",  # dirty violet
    "S020": "#EABBEB",  # pink
    "S021": "#9494E4",  # blue grey
    "S022": "#8FC787",  # eucalyptus
    "S023": "#70D6B6",  # mint
    "S024": "#DDC0A3",  # tan
}

SHOW_COLORS = {"Wile E. Coyote": "#FFAB66", "Pink Panther": "#FF66FF", "Tom & Jerry": "#7EC6E7"}

PER_JOINT_GREYS = {"HeadFront": "0.75",  # light grey
    "HandInLeft": "0.55",  # medium grey
    "HandInRight": "0.35",  # dark grey
    "Average of all joints": "0.00",  # black
}

# Joints
ALL_JOINTS = ["HeadTop", "HeadFront", "HeadRight", "HeadLeft", "SpineTop", "Chest", "ShoulderTopRight",
              "ShoulderBackRight", "ArmRight", "BackRight", "ElbowRight", "ShoulderTopLeft", "ShoulderBackLeft",
              "ArmLeft", "BackLeft", "ElbowLeft", "WristOutRight", "WristInRight", "HandOutRight", "HandInRight",
              "WristOutLeft", "WristInLeft", "HandOutLeft", "HandInLeft", "WaistBackRight", "WaistBackLeft",
              "WaistFrontRight", "WaistFrontLeft", "ThighRight", "KneeRight", "ShinRight", "AnkleRight", "HeelRight",
              "ThighLeft", "KneeLeft", "ShinLeft", "AnkleLeft", "HeelLeft", "ForefootOutRight", "ForefootInRight",
              "ForefootOutLeft", "ForefootInLeft", "ToetipRight", "ToetipLeft"]

UPPER_BODY_JOINTS = ["HeadTop", "HeadFront", "HeadRight", "HeadLeft", "SpineTop", "Chest", "ShoulderTopRight",
                     "ShoulderBackRight", "ArmRight", "BackRight", "ElbowRight", "ShoulderTopLeft", "ShoulderBackLeft",
                     "ArmLeft", "BackLeft", "ElbowLeft", "WristOutRight", "WristInRight", "HandOutRight", "HandInRight",
                     "WristOutLeft", "WristInLeft", "HandOutLeft", "HandInLeft", "WaistBackRight", "WaistBackLeft",
                     "WaistFrontRight", "WaistFrontLeft"]

BILATERAL_PAIRS = [("HeadRight", "HeadLeft"), ("HandInRight", "HandInLeft"), ("HandOutRight", "HandOutLeft"),
    ("WristInRight", "WristInLeft"), ("WristOutRight", "WristOutLeft"), ("ElbowRight", "ElbowLeft"),
    ("ArmRight", "ArmLeft"), ("ShoulderTopRight", "ShoulderTopLeft"), ("ShoulderBackRight", "ShoulderBackLeft"),
    ("BackRight", "BackLeft"), ("WaistBackRight", "WaistBackLeft"), ("WaistFrontRight", "WaistFrontLeft"),
    ("ThighRight", "ThighLeft"), ("KneeRight", "KneeLeft"), ("ShinRight", "ShinLeft"), ("HeelRight", "HeelLeft"),
    ("AnkleRight", "AnkleLeft"), ("ForefootInRight", "ForefootInLeft"), ("ForefootOutRight", "ForefootOutLeft"),
    ("ToetipRight", "ToetipLeft")]

BILATERAL_PAIRS_UPPER = [("HandInRight", "HandInLeft"), ("HandOutRight", "HandOutLeft"),
    ("WristInRight", "WristInLeft"), ("WristOutRight", "WristOutLeft"), ("ElbowRight", "ElbowLeft"),
    ("ArmRight", "ArmLeft"), ("ShoulderTopRight", "ShoulderTopLeft"), ("ShoulderBackRight", "ShoulderBackLeft")]

JOINT_GROUPS = {"Head": ["HeadRight", "HeadFront", "HeadTop", "HeadLeft"],
    "Trunk": ["SpineTop", "Chest", "BackRight", "BackLeft", "WaistBackRight", "WaistBackLeft", "WaistFrontRight",
              "WaistFrontLeft"],
    "Upper limb": ["ShoulderBackLeft", "ShoulderTopLeft", "ArmLeft", "ElbowLeft", "WristInLeft", "WristOutLeft",
                   "HandInLeft", "HandOutLeft", "ShoulderBackRight", "ShoulderTopRight", "ArmRight", "ElbowRight",
                   "WristInRight", "WristOutRight", "HandInRight", "HandOutRight"],
    "Lower limb": ["ThighLeft", "KneeLeft", "ShinLeft", "HeelLeft", "AnkleLeft", "ForefootInLeft", "ForefootOutLeft",
                   "ToetipLeft", "ThighRight", "KneeRight", "ShinRight", "HeelRight", "AnkleRight", "ForefootInRight",
                   "ForefootOutRight", "ToetipRight"], }

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

layout_up_joints = [[None, None, None, "HeadTop", None, None, None],
                           [None, None, "HeadRight", "HeadFront", "HeadLeft", None, None],
                           ["ElbowRight", "ArmRight", "ShoulderTopRight", "SpineTop", "ShoulderTopLeft", "ArmLeft", "ElbowLeft"],
                           ["WristOutRight", None, "ShoulderBackRight", None, "ShoulderBackLeft", None, "WristOutLeft"],
                           ["WristInRight", None, None, "Chest", None, None, "WristInLeft"],
                           ["HandOutRight", None, "BackRight", None, "BackLeft", None, "HandOutLeft"],
                           ["HandInRight", None, None, None, None, None, "HandInLeft"]]