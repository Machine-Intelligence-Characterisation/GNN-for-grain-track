import os
import warnings

import numpy as np
from numpy.linalg import norm

from zmatch.crystal._geo import laue_symmetries, slip_systems
# from core import defaults
# from core.quat import Quat


class Phase(object):
    laue_group_keys = {
        11 : 'cubic',
        9 : 'hexagonal',
    }

    space_group_keys = {
        225: 'fcc',
        229: 'bcc',
    }

    def __init__(self):
        """
        Parameters
        ----------
        name : str
            Name of the phase
        laue_group : int
            Laue group
        space_group : int
            Space group
        lattice_params : tuple
            Lattice parameters in order (a,b,c,alpha,beta,gamma)

        """
        self.name = None
        self.laue_group = None
        self.space_group = None
        self.lattice_params = None

        self.symmetries = None

    def load_base_information(self, name, laue_group, space_group, lattice_params=None):
        self.name = name
        self.laue_group = laue_group
        self.space_group = space_group
        self.lattice_params = lattice_params

        self.load_symmetry()
        self.load_slip_systems()

    def load_symmetry(self):
        try:
            self.symmetries = laue_symmetries[self.laue_group_keys[self.laue_group]]
        except KeyError:
            warnings.warn(f"Laue group key {self.laue_group} has not been implemented yet.", UserWarning)
            self.symmetries = None

    def load_slip_systems(self):
        try:
            self.slip_systems = slip_systems[self.space_group_keys[self.space_group]]
        except KeyError:
            warnings.warn(f"Space group key {self.space_group} has not been implemented yet.", UserWarning)
            self.slip_systems = None

