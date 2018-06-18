import numpy as np
from catlearn.optimize.constraints import apply_mask_ase_constraints
from ase.calculators.calculator import Calculator, all_changes


class CatLearnASE(Calculator):

    """Artificial CatLearn/ASE calculator.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, trained_process, ml_calc, index_constraints,
                 finite_step=1e-4, **kwargs):

        Calculator.__init__(self, **kwargs)

        self.trained_process = trained_process
        self.ml_calc = ml_calc
        self.finite_step = finite_step
        self.ind_constraints = index_constraints

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms

        # Cleaning.
        predictions = 0.0
        pred_value = 0.0
        forces = 0.0
        gradients = 0.0
        energy = 0.0

        def pred_energy_test(test, ml_calc=self.ml_calc,
                             trained_process=self.trained_process):
            # Cleaning.
            predictions = 0.0
            pred_value = 0.0

            # Get predictions.
            predictions = ml_calc.get_predictions(trained_process,
                                                  test_data=test[0])
            pred_mean = predictions['pred_mean'][0][0]
            uncertainty = predictions['uncertainty_with_reg'][0]
            pred_value = pred_mean + 4.0 * uncertainty
            return [pred_value, uncertainty]

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
            pos = test_point.copy()
            pos[0][i] = pos_flatten[index_force] + self.finite_step
            f_pos = pred_energy_test(test=pos)[0]
            pos = test_point.copy()
            pos[0][i] = pos_flatten[index_force] - self.finite_step
            f_neg = pred_energy_test(test=pos)[0]
            pos = test_point.copy()
            pos[0][i] = pos_flatten[index_force] + 2.0 * self.finite_step
            f_pos2 = pred_energy_test(test=pos)[0]
            pos = test_point.copy()
            pos[0][i] = pos_flatten[index_force] - 2.0 * self.finite_step
            f_neg2 = pred_energy_test(test=pos)[0]
            gradients[index_force] = (f_neg2 - 8.0 * f_neg + 8.0 * f_pos -
                                      f_pos2) / (12.0 * self.finite_step)

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces