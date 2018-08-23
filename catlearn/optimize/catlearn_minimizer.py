import numpy as np
from catlearn.optimize.gp_calculator import GPCalculator
from catlearn.optimize.warnings import *
from catlearn.optimize.constraints import create_mask_ase_constraints, \
                                          apply_mask_ase_constraints, unmask_geometry
from catlearn.optimize.get_real_values import get_energy_catlearn, get_forces_catlearn
from ase.data import covalent_radii
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.optimize_ml import optimize_ml_using_scipy
from catlearn.optimize.catlearn_ase_calc import CatLearnASE
from catlearn.optimize.convergence import converged
import re
from ase.atoms import Atoms
from ase.io.trajectory import TrajectoryWriter
from catlearn.optimize.io import print_info, print_info_ml, store_results, array_to_ase

class CatLearnMinimizer(object):

    def __init__(self, x0, ml_calc=None, ase_calc=None, f=None, jac=True,
                 mode='min', filename='results'):

        """Optimization setup.

        Parameters
        ----------
        x0 : ndarray
            Initial guess.
        ml_calc : ML calculator object
            Machine Learning calculator (e.g. Gaussian Processes).
        ase_calc: ASE calculator object
            When using ASE the user must pass an ASE calculator.
        f : (Optional) class (see functions.py)
            Objective function to be optimized.
        jac : boolean
            Evaluate the Jacobian (first derivatives) of the function.
        mode: string ('min' or 'max')
            User must chose if it is a minimization or maximization problem.
            Default is 'min' (Minimization).
        filename: string
            Filename to store the output.
        """

        self.jac = jac
        self.mode = mode
        self.ase = False
        self.filename = filename

        self.ml_calc = ml_calc

        if self.ml_calc is None:
            kdict = {'k1':
                         {'type': 'gaussian',
                          'scaling': 1.0, 'scaling_bounds': ((1.0, 1.0), ),
                          'width': 0.5, 'dimension':'single',
                          'bounds': ((0.01, 1.0),),},
                     'k2':
                         {'type':
                          'constant',
                          'const': 0.0}
                     }
            self.ml_calc = GPCalculator(kernel_dict=kdict,
                                        calc_uncertainty=False,
                                        guess_hyper='constant')
            warning_kernel()

        # Restart variables each run:
        self.ase_calc = ase_calc

        backup_old_calcs(self.filename)

        assert x0 is not None, err_not_x0()

        if isinstance(x0, np.ndarray):
            self.start_mode = 'array'
            self.ase_ini = None
            self.x0 = np.array(x0)
            self.fun_avail = implemented_functions()
            assert f, err_not_real_func(self)
            assert f in self.fun_avail, err_not_real_func_2(self)
            self.fun = eval(f)()
            self.list_train = [self.x0]
            self.list_targets = []
            self.list_gradients = []

        if isinstance(x0, dict):
            self.start_mode = 'dict'
            warning_prev_data_intro()
            self.list_train = np.array(x0['train'])
            assert len(self.list_train) > 1, err_train_data()
            self.list_targets = np.array(x0['targets'])
            assert len(self.list_targets) > 1, err_target_data()
            if self.jac is True:
                self.list_gradients = np.array(x0['gradients'])
                assert len(self.list_gradients) > 1, err_gradients_data()
                assert len(self.list_gradients) == len(self.list_train), \
                           err_diff_data_size()
            assert len(self.list_targets) == len(self.list_train), \
                      err_diff_data_size()
            self.fun_avail = implemented_functions()
            assert f, err_not_real_func(self)
            assert f in self.fun_avail, err_not_real_func_2(self)
            self.fun = eval(f)()

        if isinstance(x0, Atoms):
            self.start_mode = 'atoms'
            self.ase = True
            warning_using_ase()
            self.ase_ini = x0
            ase_calc_set = self.ase_calc
            self.ase_calc = self.ase_ini.get_calculator()
            if ase_calc_set is not None:
                self.ase_calc = ase_calc_set
            assert self.ase_calc, err_not_ase_calc_atoms()
            self.constraints = self.ase_ini._get_constraints()
            self.x0 = self.ase_ini.get_positions().flatten()
            self.num_atoms = self.ase_ini.get_number_of_atoms()
            self.formula = self.ase_ini.get_chemical_formula()
            self.list_train = [self.x0]
            self.list_targets = []
            self.list_gradients = []
            if len(self.constraints) < 0:
                self.constraints = None
            if self.constraints is not None:
                self.ind_mask_constr = create_mask_ase_constraints(
                self.ase_ini, self.constraints)

        if isinstance(x0, str):
            self.start_mode = 'trajectory'
            self.ase = True
            self.ase_calc = ase_calc
            self.i_ase_step = 'Previous calc.'
            warning_traj_detected()
            assert self.ase_calc, err_not_ase_calc_traj()
            trj = ase_traj_to_catlearn(traj_file=x0, ase_calc=self.ase_calc)
            self.list_train, self.list_targets, self.list_gradients, \
                trj_images, self.constraints, self.num_atoms = [
                trj['list_train'], trj['list_targets'],
                trj['list_gradients'], trj['images'],
                trj['constraints'], trj['num_atoms']]
            for i in range(1, len(trj_images)):
                self.ase_ini =  trj_images[i]
                molec_writer = TrajectoryWriter('./' + str(self.filename) +
                                                '_catlearn.traj', mode='a')
                molec_writer.write(self.ase_ini)
            if len(self.constraints) < 0:
                self.constraints = None
            if self.constraints is not None:
                self.ind_mask_constr = create_mask_ase_constraints(
                self.ase_ini, self.constraints)


    def run(self, fmax=1e-2, e_max=1e-15, max_iter=1000, min_iter=None,
            max_step=0.2, min_step=None, i_step=None,
            i_ase_step='SciPyFminCG',
            ml_algo='CG', max_memory=50):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            The convergence criterion is that the force on all individual
            atoms should be less than fmax. If not using Atoms objects or
            trajectory files (ASE) the convergence criterion is set to the
            max absolute force of each dimension.
        e_max : float
            Convergence is achieved when the energies between the last
            step and the previous step is less than e_max.
            This criterion exclusively applies when the forces
            are not evaluated.
        max_iter : float
            Maximum number of iterations before breaking the optimization
            cycle.
        min_iter : float
            Minimum number of iterations before breaking the optimization
            cycle.
        max_step : float
            Maximum distance between the next training point and the nearest
            previously trained point (in Euclidean space). Defines a
            cut-off distance of the test points w.r.t the previously trained
            data before applying a penalty to the predicted mean.
        min_step : float
            Minimum distance between the current test point and the nearest
            previously trained point (in Euclidean space). Defines a
            cut-off distance of the test points w.r.t the previously trained
            data before applying a penalty to the predicted function.
        i_step : float
            Initial step size. This applies for initializing an optimization
            for non atomistic systems. See 'i_ase_step' for atomistic systems.
        i_ase_step : string
            The first (evaluation) and second steps of the optimization will be
            performed using an ASE optimizer.
            Implemented: 'FIRE', 'BFGS', 'LBFGS' or 'MDMin'.
        ml_algo : string
            Algorithm for the optimization of the predicted mean.
            Implemented: 'L-BFGS-B', 'BFGS', 'CG',
            'Nelder-Mead', 'Powell'. Also the user can pick some algorithms
            included in ASE for atoms structure optimization, such as
            'LBFGS_ASE', 'BFGS_ASE', 'FIRE_ASE', 'MDMin_ASE' (these only apply
            when treating atoms objects and trajectory files using ASE).

        Returns
        -------
        The results of the optimization are printed in filename.txt (
        results.txt, if the user don't specify a name).
        If atoms have been introduced (ASE) a trajectory file (filename.traj)
        containing the geometries of the system for each step
        will be generated. A dictionary with a summary of the results can be
        accessed by self.results and it is also printed in 'filename'_data.txt.

        """

        self.fmax = fmax
        self.iter = 0
        self.feval = len(self.list_targets)
        self.max_step = max_step
        self.min_step = min_step
        self.i_step = i_step
        self.e_max = e_max
        self.min_iter = min_iter
        self.i_ase_step = i_ase_step
        self.ml_algo = ml_algo
        max_memory = max_memory
        self.max_iter = max_iter

        if max_memory is None:
            max_memory = 100
            if len(self.ind_mask_constr) > 100:
                max_memory = 50

        if self.i_step is None:
            self.i_step = 1e-4

        if self.ml_algo is None:
            self.ml_algo = 'Powell'
            warning_ml_algo(self)

        if self.ase is True and self.i_ase_step is None:
            self.i_ase_step = 'SciPyFminCG'

        if self.i_ase_step is not None:
            warning_first_step_ase(self)
            assert self.ase, err_first_step_ase()

        initialize(self)  # Runs exclusively in 1st and 2nd step.

        if self.ase is True and self.max_step is None:
            atomic_numbers = np.unique(self.ase_ini.numbers)
            list_radii = []
            for i in atomic_numbers:
                list_radii.append(covalent_radii[i])
            self.max_step = np.min(list_radii)/2.0

            warning_max_step_radii(max_step=self.max_step)

        if self.max_step is None:
            warning_max_step()

        # Optimization cycle:

        while not converged(self):

            initialize(self)
            converged(self)

            org_train = self.list_train.copy()
            org_targets = self.list_targets.copy()
            if self.jac is True:
                org_gradients = self.list_gradients.copy()

            if self.constraints is not None:
                [org_train, self.list_train] = apply_mask_ase_constraints(
                                           list_to_mask=self.list_train,
                                           mask_index=self.ind_mask_constr)
                [org_gradients, self.list_gradients] = \
                                           apply_mask_ase_constraints(
                                           list_to_mask=self.list_gradients,
                                           mask_index= self.ind_mask_constr)

            ############ Memory ################################
            if len(self.list_train) >= max_memory:
                self.list_train = self.list_train[-max_memory:]
                self.list_targets = self.list_targets[-max_memory:]
                self.list_gradients = self.list_gradients[-max_memory:]
            ############ Memory ################################


            # 1) Train a new process:

            # Scale the targets:
            if self.ml_calc.__dict__['scale_data'] is False:
                self.list_targets = self.list_targets - np.mean(
                                                        self.list_targets)

            if self.jac is False:
                self.list_gradients = None
            self.trained_process = self.ml_calc.train_process(
                    train_data=self.list_train,
                    target_data=self.list_targets,
                    gradients_data=self.list_gradients)


            if self.ml_calc.__dict__['guess_hyper'] is not None:
                self.trained_process = self.ml_calc.update_hyperparameters(
                    trained_process=self.trained_process)

            ########## UNDER TEST #####################################
            if self.list_fmax[-1] <= 0.10:
                self.ml_calc.__dict__['opt_hyperparam'] = True
            if self.list_fmax[-1] > 0.10:
                self.ml_calc.__dict__['opt_hyperparam'] = False
            ########## UNDER TEST #####################################


            if self.ml_calc.__dict__['opt_hyperparam']:
                self.ml_calc.opt_hyperparameters()

            # 2) Optimize ML and return the next point to evaluate:

            test0 = self.list_train[np.argmin(self.list_targets)]

            self.ase_optimizers = ['BFGS_ASE', 'LBFGS_ASE', 'FIRE_ASE',
                                   'MDMin_ASE', 'SciPyFminCG']
            self.scipy_optimizers = ['Powell', 'L-BFGS-B', 'BFGS', 'Nelder-Mead',
                                     'CG']

            assert self.ml_algo in self.scipy_optimizers or \
            self.ml_algo in self.ase_optimizers, err_not_ml_algo(self)

            if self.ml_algo in self.scipy_optimizers:
                warning_ml_algo(self)
                self.interesting_point = optimize_ml_using_scipy(self, x0=test0)

                if self.constraints is not None:
                    self.interesting_point = unmask_geometry(org_train,
                    self.interesting_point,
                    self.ind_mask_constr)
                print_info_ml(self)

            if self.ml_algo in self.ase_optimizers:
                assert self.ase, err_not_ase_algoml()
                max_ml_steps = 1000 # Hard-coded.
                ml_algo_i = re.sub('\_ASE$', '', self.ml_algo)
                warning_ml_algo(self)
                ml_ase_calc = CatLearnASE(
                                          trained_process=self.trained_process,
                                          ml_calc=self.ml_calc,
                                          index_constraints=self.ind_mask_constr,
                                          list_train=self.list_train,
                                          max_step=self.max_step,
                                          finite_step=1e-4)
                start_guess_ml = array_to_ase(input_array=(unmask_geometry(
                    org_list=org_train, masked_geom=test0,
                    mask_index=self.ind_mask_constr)),
                    num_atoms=self.num_atoms)
                start_guess_ml = Atoms(self.ase_ini, positions=start_guess_ml,
                    calculator=ml_ase_calc)
                res_min_ml = eval(ml_algo_i)(start_guess_ml, logfile=str(
                    self.filename)+ '_ASE_opt')
                res_min_ml.run(fmax=1e-3, steps=max_ml_steps) # Hard-coded.
                self.interesting_point = np.array([(
                                    start_guess_ml.get_positions()).flatten()])
                self.ml_feval_pred_mean = res_min_ml.__dict__['nsteps']
                self.ml_f_min_pred_mean = start_guess_ml.get_potential_energy()
                self.ml_convergence = False
                if self.ml_feval_pred_mean < max_ml_steps:
                    self.ml_convergence = True
                print_info_ml(self)

            self.list_train = org_train.copy()
            self.list_targets = org_targets.copy() # Rescale the targets
            if self.jac is True:
                self.list_gradients = org_gradients.copy()

            """ Break if max number of iterations is zero. This allows the 
            user to obtain the next suggested point to train (cheap) 
            without evaluating it (expensive)."""

            if self.max_iter == 0:
                break

            # 3) Add a new training point and evaluate it.

            if self.interesting_point.ndim == 1:
                self.interesting_point = np.array([self.interesting_point])

            self.list_train = np.append(self.list_train,
                                        self.interesting_point, axis=0)
            self.list_targets = np.append(self.list_targets,
                                          get_energy_catlearn(self))
            self.list_targets = np.reshape(self.list_targets,
                                          (len(self.list_targets), 1))

            if self.jac is True:
                self.list_gradients = np.append(self.list_gradients,
                [-get_forces_catlearn(self).flatten()], axis=0)

            if self.ase:
                molec_writer = TrajectoryWriter('./'+ str(
                self.filename)+ '_catlearn.traj', mode='a')
                molec_writer.write(self.ase_ini)

            # Break if reaches the max number of iterations set by the user.
            if self.max_iter <= self.iter:
                warning_max_iter_reached()
                break
            self.iter += 1
            converged(self)
            self.feval = len(self.list_targets)
            print_info(self)
            store_results(self)


def initialize(self):
    """Evaluates the "real" function for a given initial guess and then it
    will obtain the function value of a second guessed
    point originated using CG theory. This function is exclusively
    called when the optimization is initialized and the user has not
    provided any trained data.
    """

    if len(self.list_targets) == 1:

        if self.jac is False:
            alpha = np.random.normal(loc=0.0, scale=self.i_step,
                                     size=np.shape(self.list_train[0]))
            ini_train = [self.list_train[-1] - alpha]

        if self.jac is True:
            alpha = self.i_step + np.zeros_like(self.list_train[0])
            ini_train = [self.list_train[-1] - alpha *
            self.list_gradients[-1]]

        if self.i_ase_step:
            opt = eval(self.i_ase_step)(self.ase_ini).run(fmax=self.fmax,
                                                          steps=1)
            ini_train = [self.ase_ini.get_positions().flatten()]

        self.list_train = np.append(self.list_train, ini_train, axis=0)
        self.list_targets = np.append(self.list_targets,
                                      get_energy_catlearn(self))
        self.list_targets = np.reshape(self.list_targets,
                                  (len(self.list_targets), 1))
        if self.jac is True:
            self.list_gradients = np.append(
                                        self.list_gradients,
                                        -get_forces_catlearn(self).flatten())
            self.list_gradients = np.reshape(self.list_gradients,
                                             (len(self.list_targets),
                                             np.shape(self.list_train)[1])
                                             )
        self.feval = len(self.list_targets)

        if self.ase:
            molec_writer = TrajectoryWriter('./' + str(self.filename) +
                                                '_catlearn.traj', mode='a')
            molec_writer.write(self.ase_ini)
        converged(self)
        print_info(self)

    if len(self.list_targets) == 0:
        self.list_targets = [np.append(self.list_targets,
                             get_energy_catlearn(self))]
        if self.jac is True:
            self.list_gradients = [np.append(self.list_gradients,
                                   -get_forces_catlearn(self).flatten())]
        self.feval = len(self.list_targets)
        if self.ase:
            molec_writer = TrajectoryWriter('./' + str(self.filename) +
                                                '_catlearn.traj', mode='a')
            molec_writer.write(self.ase_ini)
        converged(self)
        print_info(self)
