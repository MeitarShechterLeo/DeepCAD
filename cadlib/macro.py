import numpy as np

ALL_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')
EXT_IDX = ALL_COMMANDS.index('Ext')

EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"] # b
EXTENT_TYPE = ["OneSideFeatureExtentType", "SymmetricFeatureExtentType", 
               "TwoSidesFeatureExtentType"] # e
NUM_EXTRUE_OPERATIONS = len(EXTRUDE_OPERATIONS)
NUM_EXTENT_TYPE = len(EXTENT_TYPE)

PAD_VAL = -1
N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
N_ARGS_PLANE = 3 # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4 # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4 # extrusion parameters: e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT

SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS)])
EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS)])

CMD_ARGS_MASK = np.array([[1, 1, 0, 0, 0, *[0]*N_ARGS_EXT],  # line
                          [1, 1, 1, 1, 0, *[0]*N_ARGS_EXT],  # arc
                          [1, 1, 0, 0, 1, *[0]*N_ARGS_EXT],  # circle
                          [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT],  # EOS
                          [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT],  # SOL
                          [*[0]*N_ARGS_SKETCH, *[1]*N_ARGS_EXT]]) # Extrude
                          
CMD_ARGS_MASK_SEPARETED = np.array([[1, 1, 0, 0, 0, *[0]*N_ARGS_EXT],  # line
                          [1, 1, 1, 0, 0, *[0]*N_ARGS_EXT],  # arc - flag is handled seperately
                          [1, 1, 0, 0, 1, *[0]*N_ARGS_EXT],  # circle
                          [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT],  # EOS
                          [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT],  # SOL
                          [*[0]*N_ARGS_SKETCH, *[1]*(N_ARGS_EXT-2), 0, 0]]) # Extrude - b and u is handled seperately
FLAG_IDX = 3 # only relevant for arc
EXTRUDE_OPERATION_IDX = N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_TRANS + 2 # only relevant for extrude
EXTENT_TYPE_IDX = N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_TRANS + 3 # only relevant for extrude

NORM_FACTOR = 0.75 # scale factor for normalization to prevent overflow during augmentation

MAX_N_EXT = 10 # maximum number of extrusion
MAX_N_LOOPS = 6 # maximum number of loops per sketch
MAX_N_CURVES = 15 # maximum number of curves per loop
MAX_TOTAL_LEN = 60 # maximum cad sequence length
ARGS_DIM = 256
