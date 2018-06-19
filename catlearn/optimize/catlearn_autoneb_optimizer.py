import numpy as np
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, store_results_neb
from catlearn.optimize.ml_calculator import GPCalculator, train_ml_process
from catlearn.optimize.convergence import get_fmax
from catlearn.optimize.get_real_values import eval_and_append
from catlearn.optimize.constraints import create_mask_ase_constraints
from catlearn.optimize.plots import get_plot_mullerbrown, get_plots_neb, \
                                    get_plot_mullerbrown_p
from ase.io.trajectory import TrajectoryWriter
from ase.neb import NEBTools
from ase.io import read, write
import copy
import os
from catlearn.optimize.autoneb_ase import AutoNEBASE
import glob
import shutil
from catlearn.optimize.catlearn_ase_calc import CatLearnASE
from scipy.spatial import distance


class AutoNEBOptimizer(object):

    def __init__(self, start, end, n_images=None,
                 ml_calc=None, ase_calc=None, inc_prev_calcs=False,
                 restart=False, spring=None):
        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file (in ASE format).
            Initial end-point of the NEB path.
        end: Trajectory file (in ASE format).
            Final end-point of the NEB path.
        n_images: int
            Number of images of the path (if not included a path before).
             The number of images include the 2 end-points of the NEB path.
        spring: float
            Spring constant(s) in eV/Ang.
        ml_calc : ML calculator Object.
            Machine Learning calculator (e.g. Gaussian Process). Default is GP.
        ase_calc: ASE calculator Object.
            ASE calculator as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
        """

        # Start end-point, final end-point and path (optional):
        self.start = start
        self.end = end
        self.n_images = n_images

        # General setup:
        self.iter = 0
        self.ml_calc = ml_calc
        self.ase_calc = ase_calc
        self.ase = True
        self.spring = spring

        # Reset:
        self.constraints = None
        self.interesting_point = None
        self.uncertainty_path = []
        self.path_distance = 0.0

        # Create new file to store warnings and errors:
        open('warnings_and_errors.txt', 'w')

        assert start is not None, err_not_neb_start()
        assert end is not None, err_not_neb_end()
        assert self.ase_calc, err_not_ase_calc_traj()

        # A) Include previous calculations for training the ML model.
        is_endpoint = read(start, ':')
        fs_endpoint = read(end, ':')

        # B) Only include initial and final (optimized) images.
        if inc_prev_calcs is False:
            is_endpoint = read(start, '-1:')
            fs_endpoint = read(end, '-1:')

        self.initial_image = read(start, '-1')
        self.final_image = read(end, '-1')

        # A and B) Restart mode.

        # Read the previously evaluated structures and append them to the
        # training list
        if restart is True:
            restart_filename = 'evaluated_structures.traj'
            if not os.path.isfile(restart_filename):
                warning_restart_neb()
            if os.path.isfile(restart_filename):
                evaluated_images = read(restart_filename, ':')
                is_endpoint = evaluated_images + is_endpoint

        # Write previous evaluated images in evaluations:
        write('./evaluated_structures.traj',
              is_endpoint + fs_endpoint)

        # Check the magnetic moments of the initial and final states:
        self.magmom_is = None
        self.magmom_fs = None
        self.magmom_is = is_endpoint[-1].get_initial_magnetic_moments()
        self.magmom_fs = fs_endpoint[-1].get_initial_magnetic_moments()

        self.spin = False
        if not np.array_equal(self.magmom_is, np.zeros_like(self.magmom_is)):
            self.spin = True
        if not np.array_equal(self.magmom_fs, np.zeros_like(self.magmom_fs)):
            self.spin = True
            warning_spin_neb()
        if np.array_equal(self.magmom_is, self.magmom_fs):
            self.spin = False

        # Obtain the energy of the endpoints for scaling:
        energy_is = is_endpoint[-1].get_potential_energy()
        energy_fs = fs_endpoint[-1].get_potential_energy()

        # Set scaling of the targets:
        self.scale_targets = np.min([energy_is, energy_fs])

        # Convert atoms information into data to feed the ML process.
        if os.path.exists('./tmp.traj'):
                os.remove('./tmp.traj')
        merged_trajectory = is_endpoint + fs_endpoint
        write('tmp.traj', merged_trajectory)

        trj = ase_traj_to_catlearn(traj_file='tmp.traj')
        self.list_train, self.list_targets, self.list_gradients, trj_images,\
            self.constraints, self.num_atoms = [trj['list_train'],
                                                trj['list_targets'],
                                                trj['list_gradients'],
                                                trj['images'],
                                                trj['constraints'],
                                                trj['num_atoms']]
        os.remove('./tmp.traj')
        self.ase_ini = trj_images[0]
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.ind_mask_constr = create_mask_ase_constraints(
                                                self.ase_ini, self.constraints)

        # Configure ML calculator.
        if self.ml_calc is None:
            self.kdict = {'k1': {'type': 'gaussian', 'width': 0.5,
                                 'dimension': 'single',
                                 'bounds': ((0.1, 1.0), ),
                                 'scaling': 1.0,
                                 'scaling_bounds': ((1.0, 1.0), )}
                          }

            self.ml_calc = GPCalculator(
                kernel_dict=self.kdict, opt_hyperparam=False, scale_data=False,
                scale_optimizer=False, calc_uncertainty=True,
                regularization=1e-5, regularization_bounds=(1e-5, 1e-5))

        # Stabilize spring constant:

        is_pos = is_endpoint[-1].get_positions().flatten()
        fs_pos = fs_endpoint[-1].get_positions().flatten()
        self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos))

        if self.spring is None:
            self.spring = np.sqrt((self.n_images-1) / self.d_start_end)

    def run(self, fmax=0.05, unc_convergence=0.010, max_iter=500,
            ml_max_iter=500, plot_neb_paths=False):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs).
        unc_convergence: float
            Maximum uncertainty for convergence (in eV).
        max_iter : int
            Maximum number of iterations in the surrogate model.
        ml_max_iter : int
            Maximum number of ML NEB iterations.
        plot_neb_paths: bool
            If True it prints and stores (in csv format) the last predicted
            NEB path obtained by the surrogate ML model. Note: Python package
            matplotlib is required.

        Returns
        -------
        NEB optimized path.
        Files :
        """

        while True:

            # 1) Train Machine Learning process:

            # Check that the user is not feeding redundant information to ML.

            count_unique = np.unique(self.list_train, return_counts=True,
                                     axis=0)[1]
            msg = 'Your training list constains 1 or more duplicated elements'
            assert np.any(count_unique) < 2, msg

            print('Training a ML process...')
            print('Number of training points:', len(self.list_targets))
            process = train_ml_process(list_train=self.list_train,
                                       list_targets=self.list_targets,
                                       list_gradients=self.list_gradients,
                                       index_constraints=self.ind_mask_constr,
                                       ml_calculator=self.ml_calc,
                                       scaling_targets=self.scale_targets)

            trained_process = process['trained_process']
            ml_calc = process['ml_calc']
            print('ML process trained.')

            # 2) Setup and run ML NEB:

            # Read initial and final:
            initial = copy.deepcopy(self.initial_image)
            final = copy.deepcopy(self.final_image)

            initial.__dict__['_calc'].__dict__['results']['energy'] = \
                initial.__dict__['_calc'].__dict__['results']['energy'] - \
                self.scale_targets
            final.__dict__['_calc'].__dict__['results']['energy'] = \
                final.__dict__['_calc'].__dict__['results']['energy'] - \
                self.scale_targets

            # Check there are no previous images:
            for i in glob.glob("images*"):
                os.remove(i)

            # Remove data from previous iteration:
            directory = './AutoNEB_iter'
            if os.path.exists(directory):
                shutil.rmtree(directory)

            write('images000.traj', initial)
            write('images001.traj', final)

            print('Starting ML NEB optimization...')
            neb_opt = AutoNEBASE(
                                 n_simul=1,
                                 n_max=self.n_images,
                                 fmax=fmax,
                                 prefix='images',
                                 attach_calculators=CatLearnASE(
                                         trained_process=trained_process,
                                         ml_calc=ml_calc,
                                         kappa=2.0,
                                         index_constraints=self.ind_mask_constr
                                         ),
                                 k=self.spring,
                                 maxsteps=ml_max_iter
                                 )
            neb_opt.run()
            print('ML NEB optimized.')

            # 3) Get results from ML NEB using ASE NEB tools:
            # See https://wiki.fysik.dtu.dk/ase/ase/neb.html

            interesting_point = []

            # Get images loading traj files:

            images = []
            for i in range(0, self.n_images):
                struc_i = read('%s%03d.traj' % ('images', i))
                struc_i.info['iteration'] = self.iter
                images.append(struc_i)

            images[0].info['uncertainty'] = 0.0
            images[-1].info['uncertainty'] = 0.0

            # Get fit of the discrete path.

            neb_tools = NEBTools(images)
            [s, e, sfit, efit] = neb_tools.get_fit()[0:4]

            self.path_distance = s[-1]

            self.uncertainty_path = []
            energies_path = []
            for i in images:
                self.uncertainty_path.append(i.info['uncertainty'])
                energies_path.append(i.get_total_energy())

            # Select image with maximum uncertainty.
            if self.iter % 2 == 0:
                argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                interesting_point = images[1:-1][
                                      argmax_unc].get_positions().flatten()

            # Select image with max. predicted value (absolute value).
            if self.iter % 2 == 1:
                argmax_unc = np.argmax(np.abs(energies_path[1:-1]))
                interesting_point = images[1:-1][
                                          int(argmax_unc)].get_positions(
                                          ).flatten()

            # Plots results in each iteration.
            if plot_neb_paths is True:
                get_plots_neb(images=images,
                              selected=int(argmax_unc), iter=self.iter)

            if plot_neb_paths is True:
                if self.ase_calc.__dict__['name'] == 'mullerbrown':
                    get_plot_mullerbrown(images=images,
                                         interesting_point=interesting_point,
                                         trained_process=trained_process,
                                         list_train=self.list_train)
                    # get_plot_mullerbrown_p(images=images,
                    #                        interesting_point=interesting_point,
                    #                        trained_process=trained_process,
                    #                        list_train=self.list_train)
            # Store results each iteration:
            store_results_neb(s, e, sfit, efit, self.uncertainty_path)

            # 3) Add a new training point and evaluate it.

            eval_and_append(self, interesting_point)

            # 4) Store results.

            # Evaluated images.
            TrajectoryWriter(atoms=self.ase_ini,
                             filename='./evaluated_structures.traj',
                             mode='a').write()
            # Last path.
            write('last_predicted_path.traj', images)

            # All paths.
            for i in images:
                TrajectoryWriter(atoms=i,
                                 filename='all_predicted_paths.traj',
                                 mode='a').write()

            print('Length of the current path (Angstrom):', self.path_distance)
            print('Spring constant (eV/Angstrom):', self.spring)
            print('Max. uncertainty (eV):',
                  np.max(self.uncertainty_path[1:-1]))
            print('Image #id with max. uncertainty:',
                  np.argmax(self.uncertainty_path[1:-1]) + 2)
            print('Number of iterations:', self.iter)

            # Break if converged:

            max_forces = get_fmax(-np.array([self.list_gradients[-1]]),
                                  self.num_atoms)
            max_abs_forces = np.max(np.abs(max_forces))

            print('Max. force of the last image evaluated (eV/Angstrom):',
                  max_abs_forces)
            print('Energy of the last image evaluated (eV):',
                  self.list_targets[-1][0])
            print('Energy of the last image evaluated w.r.t. to endpoint ('
                  'eV):', self.list_targets[-1][0] - self.scale_targets)
            print('Number #id of the last evaluated image:', argmax_unc + 2)

            # 5) Check convergence:

            # Check whether the evaluated point is a stationary point.
            if max_abs_forces <= fmax:
                congrats_stationary_neb()
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    congrats_neb_converged()
                    break

            # Break if reaches the max number of iterations set by the user.
            if max_iter <= self.iter:
                warning_max_iter_reached()
                break

        # Print Final convergence:
        print('Number of function evaluations in this run:', self.iter)