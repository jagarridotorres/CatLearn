import numpy as np
from catlearn.optimize.constraints import apply_mask_ase_constraints
import gptools


def train_ml_process(list_train, list_targets, list_gradients,
                     index_constraints, ml_calculator, scaling_targets,
                     opt_hyper):
    """Trains a machine learning process.

    Parameters (self):

    Parameters
    ----------
    list_train : ndarray
        List of positions (in Cartesian).
    list_targets : ndarray
        List of energies.
    list_gradients : ndarray
        List of gradients.
    index_constraints : ndarray
        List of constraints constraints generated
        previously. In order to 'hide' fixed atoms to the ML
        algorithm we create a constraints mask. This
        allows to reduce the size of the training
        features (avoids creating a large covariance matrix).
    ml_calculator: object
        Machine learning calculator. See above.
    scaling_targets: float
        Scaling of the train targets.
    Returns
    --------
    dictionary containing:
        scaling_targets : scaling for the energies of the training set.
        trained_process : trained process ready for testing.
        ml_calc : returns the ML calculator (if changes have been made,
              e.g. hyperparamter optimization).


    """

    if index_constraints is not None:
        list_train = apply_mask_ase_constraints(
                                   list_to_mask=list_train,
                                   mask_index=index_constraints)[1]
        list_gradients = apply_mask_ase_constraints(
                                        list_to_mask=list_gradients,
                                        mask_index=index_constraints)[1]

    # Scale energies.

    list_targets = list_targets - np.ones_like(list_targets) * scaling_targets

    list_targets = list_targets.flatten()


    ml_calculator.add_data(list_train, list_targets.flatten(),
                                             err_y=1e-6)
    for i in range(0, np.shape(list_gradients)[1]):
        g_i = list_gradients[:, i]
        n_i = np.zeros(np.shape(list_gradients))
        n_i[:, i] = 1.0
        ml_calculator.add_data(list_train, g_i, n=n_i, err_y=1e-6)

    if opt_hyper is True:
        ml_calculator.sample_hyperparameter_posterior(nsamp=5)
    return ml_calculator
