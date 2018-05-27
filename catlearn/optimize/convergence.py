import numpy as np
from catlearn.optimize.get_real_values import *
from catlearn.optimize.warnings import *
from catlearn.optimize.constraints import *
from catlearn.optimize.io import *

def get_fmax(gradients_flatten, num_atoms):
    """Function that print a list of max. individual atom forces."""

    list_fmax = np.zeros((len(gradients_flatten), 1))
    j = 0
    for i in gradients_flatten:
        atoms_forces_i = np.reshape(i, (num_atoms, 3))
        list_fmax[j] = np.max(np.sqrt(np.sum(atoms_forces_i**2, axis=1)))
        j = j + 1
    return list_fmax


def converged(self):
    """Function that checks the convergence in each optimization step."""

    #  The force on all individual atoms should be less than fmax.
    if self.ase is True:
        if self.jac is True:
            self.list_fmax = get_fmax(self.list_gradients, self.num_atoms)
            self.max_abs_forces = self.list_fmax[-1][0]
            if self.min_iter:
                if self.iter <= self.min_iter:
                    return False
            if self.max_abs_forces < self.fmax:
                return True

    # The force on all individual components should be less than fmax.
    if self.ase is False:
        if self.jac is True:
            self.list_fmax = np.amax(np.abs(self.list_gradients),axis=1)
            forces_last_iteration = -self.list_fmax[-1]
            self.max_abs_forces = np.max(np.abs(forces_last_iteration))
            if self.min_iter:
                if self.iter <= self.min_iter:
                    return False
            if self.max_abs_forces < self.fmax:
                return True

    # Check energy convergence.
    if self.feval > 1:
        self.e_diff = np.abs(self.list_targets[-1] - self.list_targets[-2])

    if self.min_iter:
        if self.iter <= self.min_iter:
            return False

    if not self.jac:
        if len(self.list_targets) > 1:
            if self.e_diff < self.e_max:
                return True

    return False


def neb_converged(self):

    if self.iter > 1:
        if np.max(self.unc_discr_neb) <= self.unc_conv:
            if self.distance_convergence <= 0.1:
                print('Path has not changed from previous iteration.')
                print('Max uncertainty:', np.max(self.unc_discr_neb))
                fmax = get_fmax(-np.array([self.list_gradients[-1]]),
                                    self.num_atoms)
                self.max_abs_forces = np.max(np.abs(fmax))
                print('Forces last image evaluated', self.max_abs_forces)
                if self.max_abs_forces <= self.fmax:
                    return True
                # Prevents to evaluate twice the same point:
                if np.argmax(self.unc_discr_neb) == np.argmax(
                                                    self.energies_discr_neb):
                    return False
                if self.max_abs_forces >= self.fmax:
                    check_point = self.images[np.argmax(
                                              self.energies_discr_neb)
                                              ].get_positions().flatten()
                    if check_point.ndim == 1:
                        check_point = np.array([check_point])
                    self.list_train = np.append(self.list_train,
                                        check_point, axis=0)
                    self.list_targets = np.append(self.list_targets,
                                          get_energy_catlearn(self))
                    self.list_targets = np.reshape(self.list_targets,
                                           (len(self.list_targets), 1))
                    self.list_gradients = np.append(self.list_gradients,
                                            [-get_forces_catlearn(
                                             self).flatten()], axis=0)
                    TrajectoryWriter(atoms=self.ase_ini, filename='./' + str(self.filename)
                     +'_evaluated_images.traj', mode='a').write()

                    self.iter += 1
                    fmax = get_fmax(-np.array([self.list_gradients[-1]]),
                                    self.num_atoms)
                    self.max_abs_forces = np.max(np.abs(fmax))
                    print('Forces max. top image', self.max_abs_forces)

                    ######### Under test: ############
                    self.ci = True
                    ######### Under test: ############
                    if self.climb_img is False:
                        print('WARNING: The path is not converged using CI.')
                        return True
                    if self.max_abs_forces > self.fmax:
                        return False
                print('Congratulations your NEB path is converged!')
                return True
    return False