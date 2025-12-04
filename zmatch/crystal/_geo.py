import numpy as np
from collections import OrderedDict  # Only needed for Python < 3.7


overRoot2 = np.sqrt(2) / 2
sqrt3over2 = np.sqrt(3) / 2
# Use ideal ratio as only used for plotting unit cell
cOverA = 1.633 / 2

# Here gives laue group symmetries
laue_symmetries = {
    "cubic": [
            # identity
            (1.0, 0.0, 0.0, 0.0),

            # cubic tetrads(100)
            (overRoot2, overRoot2, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (overRoot2, -overRoot2, 0.0, 0.0),
            (overRoot2, 0.0, overRoot2, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (overRoot2, 0.0, -overRoot2, 0.0),
            (overRoot2, 0.0, 0.0, overRoot2),
            (0.0, 0.0, 0.0, 1.0),
            (overRoot2, 0.0, 0.0, -overRoot2),

            # cubic dyads (110)
            (0.0, overRoot2, overRoot2, 0.0),
            (0.0, -overRoot2, overRoot2, 0.0),
            (0.0, overRoot2, 0.0, overRoot2),
            (0.0, -overRoot2, 0.0, overRoot2),
            (0.0, 0.0, overRoot2, overRoot2),
            (0.0, 0.0, -overRoot2, overRoot2),

            # cubic triads (111)
            (0.5, 0.5, 0.5, 0.5),
            (0.5, -0.5, -0.5, -0.5),
            (0.5, -0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5, -0.5),
            (0.5, 0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5, 0.5)
            ],

    "hexagonal":
        [
            # identity
            (1.0, 0.0, 0.0, 0.0),

            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),

            # hexagonal hexads
            (sqrt3over2, 0.0, 0.0, 0.5),
            (0.5, 0.0, 0.0, sqrt3over2),
            (0.5, 0.0, 0.0, -sqrt3over2),
            (sqrt3over2, 0.0, 0.0, -0.5),

            # hexagonal diads
            (0.0, -0.5, -sqrt3over2, 0.0),
            (0.0, 0.5, -sqrt3over2, 0.0),
            (0.0, sqrt3over2, -0.5, 0.0),
            (0.0, -sqrt3over2, -0.5, 0.0)
        ]
}

slip_systems = {
    'fcc': {'all': {
        (1, 1, 1): [(0, 1, -1), (-1, 0, 1), (1, -1, 0)],
        (1, 1, -1): [(0, 1, 1), (-1, 0, -1), (1, -1, 0)],
        (-1, 1, 1): [(0, 1, -1), (1, 0, 1), (-1, -1, 0)],
        (1, -1, 1): [(0, -1, -1), (-1, 0, 1), (1, 1, 0)]
    }
    }
}
