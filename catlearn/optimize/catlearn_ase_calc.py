import numpy as np
from catlearn.optimize.constraints import apply_mask_ase_constraints
from ase.calculators.calculator import Calculator, all_changes
import copy


class CatLearnASE(Calculator):

    """Artificial CatLearn/ASE calculator.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, ml_calc, index_constraints,
                 finite_step=1e-5, kappa=0.0, **kwargs):

        Calculator.__init__(self, **kwargs)

        self.ml_calc = ml_calc
        self.fs = finite_step
        self.ind_constraints = index_constraints
        self.kappa = kappa

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms

        # Clean up:
        energy = 0.0
        forces = 0.0
        uncertainty = 0.0

        def pred_energy_test(test, ml_calc=self.ml_calc,
                             kappa=self.kappa):
            # Get predictions.
            post_mean = 0.0
            unc = 0.0
            post_mean, unc = ml_calc.predict(test)
            acq_val = copy.deepcopy(post_mean) + (kappa * copy.deepcopy(unc))
            return [acq_val, unc]

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()

        test_point = apply_mask_ase_constraints(
                                            list_to_mask=[pos_flatten],
                                            mask_index=self.ind_constraints)[1]

        # Get energy.
        energy = pred_energy_test(test=test_point)[0]

        # Get uncertainty.
        uncertainty = pred_energy_test(test=test_point)[1]

        # Attach uncertainty to Atoms object.
        atoms.info['uncertainty'] = uncertainty

        # Get forces:
        gradients = np.zeros(len(pos_flatten))
        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
            pos = copy.deepcopy(test_point)
            pos[0][i] = pos_flatten[index_force] + self.fs
            f_pos = pred_energy_test(test=pos)[0]
            pos = copy.deepcopy(test_point)
            pos[0][i] = pos_flatten[index_force] - self.fs
            f_neg = pred_energy_test(test=pos)[0]
            gradients[index_force] = (-f_neg + f_pos) / (2.0 * self.fs)

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
