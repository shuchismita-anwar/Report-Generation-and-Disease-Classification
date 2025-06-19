ANATOMICAL_REGIONS = {
    'cardiac_silhouette': {
        'name': 'cardiac silhouette',
        'keywords': ['cardiac silhouette'],
        'bounds': (0.37321170951463667, 0.47769894973347626, 0.7700050224868392, 0.7546451428797857),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.935)
    },
    'left_clavicle': {
        'name': 'left clavicle',
        'keywords': ['left clavicle'],
        'bounds': (0.5572333185927673, 0.11955842905079471, 0.9437893081761006, 0.2684399550110119),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2188 valid samples (EyeGaze: 1075, REFLACX: 1113, 5 filtered, avg_conf: 0.905)
    },
    'left_costophrenic_angle': {
        'name': 'left costophrenic angle',
        'keywords': ['left costophrenic angle'],
        'bounds': (0.8394839261483314, 0.7251755651215535, 0.9399232169009046, 0.8168640451570541),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 1982 valid samples (EyeGaze: 869, REFLACX: 1113, 214 filtered, avg_conf: 0.805)
    },
    'left_hilar_structures': {
        'name': 'left hilar structures',
        'keywords': ['left hilar structures'],
        'bounds': (0.5300927893048408, 0.35379003634527917, 0.6969339622641509, 0.5667539267015707),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.934)
    },
    'left_lower_lung_zone': {
        'name': 'left lower lung zone',
        'keywords': ['left lower lung zone'],
        'bounds': (0.5365964561733096, 0.5356675392670157, 0.8978004952454568, 0.8021890555254403),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2160 valid samples (EyeGaze: 1047, REFLACX: 1113, 36 filtered, avg_conf: 0.929)
    },
    'left_lung': {
        'name': 'left lung',
        'keywords': ['left lung'],
        'bounds': (0.5259046105190097, 0.12478328308312486, 0.8980233455747089, 0.8024001120456901),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2158 valid samples (EyeGaze: 1045, REFLACX: 1113, 38 filtered, avg_conf: 0.933)
    },
    'left_mid_lung_zone': {
        'name': 'left mid lung zone',
        'keywords': ['left mid lung zone'],
        'bounds': (0.5349430477518289, 0.38631738792539266, 0.8632075471698113, 0.5384190468634817),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2190 valid samples (EyeGaze: 1077, REFLACX: 1113, 6 filtered, avg_conf: 0.932)
    },
    'left_upper_lung_zone': {
        'name': 'left upper lung zone',
        'keywords': ['left upper lung zone'],
        'bounds': (0.5312062408550223, 0.14465408805031446, 0.8372934968385697, 0.3868026013644237),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2194 valid samples (EyeGaze: 1081, REFLACX: 1113, 2 filtered, avg_conf: 0.942)
    },
    'right_clavicle': {
        'name': 'right clavicle',
        'keywords': ['right clavicle'],
        'bounds': (0.05905511811023622, 0.11811023622047244, 0.451380065598413, 0.2718996011985923),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2193 valid samples (EyeGaze: 1080, REFLACX: 1113, avg_conf: 0.917)
    },
    'right_costophrenic_angle': {
        'name': 'right costophrenic angle',
        'keywords': ['right costophrenic angle'],
        'bounds': (0.0479559748427673, 0.7185534591194969, 0.14595459470742939, 0.8091084345473045),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2146 valid samples (EyeGaze: 1033, REFLACX: 1113, 50 filtered, avg_conf: 0.849)
    },
    'right_hilar_structures': {
        'name': 'right hilar structures',
        'keywords': ['right hilar structures'],
        'bounds': (0.29450261780104714, 0.3547833202154841, 0.46156007958189293, 0.5625),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.937)
    },
    'right_lower_lung_zone': {
        'name': 'right lower lung zone',
        'keywords': ['right lower lung zone'],
        'bounds': (0.09630503144654089, 0.5313971408250037, 0.44175392670157065, 0.7899214659685864),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2166 valid samples (EyeGaze: 1053, REFLACX: 1113, 30 filtered, avg_conf: 0.938)
    },
    'right_lung': {
        'name': 'right lung',
        'keywords': ['right lung'],
        'bounds': (0.09562001138363245, 0.125, 0.4706829179696034, 0.7906659323645568),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2166 valid samples (EyeGaze: 1053, REFLACX: 1113, 30 filtered, avg_conf: 0.958)
    },
    'right_mid_lung_zone': {
        'name': 'right mid lung zone',
        'keywords': ['right mid lung zone'],
        'bounds': (0.13551193485070873, 0.38691391098177536, 0.45558176100628933, 0.5350437345322385),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.942)
    },
    'right_upper_lung_zone': {
        'name': 'right upper lung zone',
        'keywords': ['right upper lung zone'],
        'bounds': (0.1608359707201128, 0.14606603744330576, 0.468688186789311, 0.3868989338077552),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.949)
    },
    'trachea': {
        'name': 'trachea',
        'keywords': ['trachea'],
        'bounds': (0.4021023414508872, 0.05791884816753927, 0.5579188481675392, 0.4904635681013669),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.852)
    },
    'upper_mediastinum': {
        'name': 'upper mediastinum',
        'keywords': ['upper mediastinum'],
        'bounds': (0.4086405080948316, 0.2153872544852851, 0.6273584905660378, 0.4739447883883056),  # (x_min, y_min, x_max, y_max) - normalized fractions
        # Statistics: 2196 valid samples (EyeGaze: 1083, REFLACX: 1113, avg_conf: 0.913)
    },
}
