import numpy as np
from scipy.spatial import distance
from catlearn.optimize.convert import *


def penalty_too_far_atoms(list_train, test, max_step, c_max_crit=1e2):
    d_test_list_train = distance.cdist([test], list_train, 'euclidean')
    closest_train = (list_train[np.argmin(d_test_list_train)])
    test = array_to_atoms(test)
    closest_train = array_to_atoms(closest_train)
    penalty = 0
    for atom in range(len(test)):
        d_atom_atom = distance.euclidean(test[atom], closest_train[atom])
        if d_atom_atom >= max_step:
            p_i = c_max_crit * (d_atom_atom-max_step)**2
        else:
            p_i = 0
        penalty += p_i
    return penalty


def penalty_too_far_atoms_v2(list_train, test, max_step, penalty_constant=1e3):
    d_test_list_train = distance.cdist([test], list_train, 'euclidean')
    closest_train = (list_train[np.argmin(d_test_list_train)])
    test = array_to_atoms(test)
    closest_train = array_to_atoms(closest_train)
    penalty = 0
    for atom in range(len(test)):
        d_atom_atom = distance.euclidean(test[atom], closest_train[atom])
        if d_atom_atom >= max_step:
            a_const = penalty_constant
            c_const = 2.0
            d_const = 1.0
            p_i = (a_const * ((d_atom_atom-max_step)**2)) / (c_const*np.abs(
            d_atom_atom-max_step) + d_const)
        else:
            p_i = 0
        penalty += p_i
    return penalty


def penalty_too_far(list_train, test, max_step=None, c_max_crit=1e2):
    """ Pass an array of test features and train features and
    returns an array of penalties due to 'too far distance'.
    This prevents to explore configurations that are unrealistic.

    Parameters
    ----------
    d_max_crit : float
        Critical distance.
    c_max_crit : float
        Constant for penalty minimum distance.
    penalty_max: array
        Array containing the penalty to add.
    """
    penalty_max = []
    for i in test:
        d_max = np.min(distance.cdist([i], list_train,'euclidean'))
        if d_max >= max_step:
            p = c_max_crit * (d_max-max_step)**2
        else:
            p = 0.0
        penalty_max.append(p)
    return penalty_max


class PenaltyFunctions(object):
    """Base class for penalty functions."""

    def __init__(self, targets=None, predictions=None, uncertainty=None,
                 train_features=None, test_features=None):
        """Initialization of class.

        Parameters
        ----------
        targets : list
            List of known target values.
        predictions : list
            List of predictions from the GP.
        uncertainty : list
            List of variance on the GP predictions.
        train_features : array
            Feature matrix for the training data.
        test_features : array
            Feature matrix for the test data.

        """
        self.targets = targets
        self.predictions = predictions
        self.uncertainty = uncertainty
        self.train_features = train_features
        self.test_features = test_features

    def penalty_close(self, c_min_crit=1e5, d_min_crit=1e-5):
        """Penalize data that is too close.

        Pass an array of test features and train features and returns an array
        of penalties due to 'too short distance' ensuring no duplicates are
        added.

        Parameters
        ----------
        d_min_crit : float
            Critical distance.
        c_min_crit : float
            Constant for penalty minimum distance.
        penalty_min: array
            Array containing the penalty to add.
        """
        penalty_min = []
        for i in self.test_features:
            d_min = np.min(
                distance.cdist([i], self.train_features, 'euclidean'))
            p = 0.0
            if d_min < d_min_crit:
                p = c_min_crit * (d_min - d_min_crit)**2
            penalty_min.append(p)

        return penalty_min

    def penalty_far(self, c_max_crit=1e2, d_max_crit=10.0):
        """Penalize data that is too far.

        Pass an array of test features and train features and returns an array
        of penalties due to 'too far distance'. This prevents to explore
        configurations that are unrealistic.

        Parameters
        ----------
        d_max_crit : float
            Critical distance.
        c_max_crit : float
            Constant for penalty minimum distance.
        penalty_max: array
            Array containing the penalty to add.
        """
        penalty_max = []
        for i in self.test_features:
            d_max = np.min(
                distance.cdist([i], self.train_features, 'euclidean'))
            p = 0.0
            if d_max > d_max_crit:
                p = c_max_crit * (d_max - d_max_crit)**2
            penalty_max.append(p)

        return penalty_max

