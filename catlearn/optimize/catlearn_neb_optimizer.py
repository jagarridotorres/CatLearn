import numpy as np
import re
from catlearn.optimize.gp_calculator import GPCalculator
from catlearn.optimize.warnings import *
from catlearn.optimize.constraints import *
from catlearn.optimize.convergence import *
from catlearn.optimize.initialize import *
from catlearn.optimize.neb_tools import *
from catlearn.optimize.neb_functions import *
from ase import Atoms
from scipy.spatial import distance
from ase.io import read, write
# from ase.visualize import view
from ase.neb import NEBTools
import copy
from catlearn.optimize.plots import plot_neb_mullerbrown, plot_predicted_neb_path
from ase.optimize import QuasiNewton, BFGS, FIRE, MDMin
import matplotlib.pyplot as plt


class NEBOptimizer(object):

    def __init__(self, start=None, end=None,
                 ml_calc=None, ase_calc=None, filename='results',
                 inc_prev_calcs=False, n_images=None,
                 interpolation='', neb_method='improvedtangent', spring=100.0):
        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file or atoms object (ASE)
            First end point of the NEB path. Starting geometry.
        end: Trajectory file or atoms object (ASE)
            Second end point of the NEB path. Final geometry.
        path: Trajectory file or atoms object (ASE)
            Atoms trajectory with the intermediate images of the path
            between the two end-points.
        ml_calc : ML calculator object
            Machine Learning calculator (e.g. Gaussian Processes).
            Default is GP.
        ase_calc: ASE calculator object
            When using ASE the user must pass an ASE calculator.
            Default is None (Max uncertainty of the NEB predicted path.)
        filename: string
            Filename to store the output.
        n_images: number of images including initial and final images (
        end-points).
        """

        # Start end-point, final end-point and path (optional):
        self.start = start
        self.end = end
        self.n_images = n_images

        # General setup:
        self.iter = 0
        self.ml_calc = ml_calc
        self.ase_calc = ase_calc
        self.filename = filename
        self.ase = True

        # Reset:
        self.constraints = None
        self.interesting_point = None

        assert start is not None, err_not_neb_start()
        assert end is not None, err_not_neb_end()

        self.ase = True
        self.ase_calc = ase_calc
        assert self.ase_calc, err_not_ase_calc_traj()

        # A) Include previous calculations for training the ML model.
        is_endpoint = read(start, ':')
        fs_endpoint = read(end, ':')


        # B) Only include initial and final (optimized) images.
        if inc_prev_calcs is False:
            is_endpoint = read(start, '-1:')
            fs_endpoint = read(end, '-1:')
        is_pos = is_endpoint[-1].get_positions().flatten()
        fs_pos = fs_endpoint[-1].get_positions().flatten()

        # Write previous evaluated images in evaluations:
        write('./' + str(self.filename) +'_evaluated_images.traj',
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

        # Obtain the energy of the endpoints for scaling:
        energy_is = is_endpoint[-1].get_potential_energy()
        energy_fs = fs_endpoint[-1].get_potential_energy()


        # Set scaling of the targets:
        self.scale_targets = np.min([energy_is, energy_fs])

        # Convert atoms information to ML information.
        if os.path.exists('./tmp.traj'):
                os.remove('./tmp.traj')
        merged_trajectory = is_endpoint + fs_endpoint
        write('tmp.traj', merged_trajectory)

        trj = ase_traj_to_catlearn(traj_file='tmp.traj',
                                ase_calc=self.ase_calc)
        self.list_train, self.list_targets, self.list_gradients, \
            trj_images, self.constraints, self.num_atoms = [
            trj['list_train'], trj['list_targets'],
            trj['list_gradients'], trj['images'],
            trj['constraints'], trj['num_atoms']]
        os.remove('./tmp.traj')
        self.ase_ini =  trj_images[0]
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.ind_mask_constr = create_mask_ase_constraints(
            self.ase_ini, self.constraints)

        # Calculate length of the path.
        self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos))

        # Configure ML.
        if self.ml_calc is None:
            self.kdict = {
                         'k1': {'type': 'gaussian', 'width': 0.5,
                         'bounds':
                          ((0.01, 1.0),)*len(self.ind_mask_constr),
                         'scaling': 1.0, 'scaling_bounds': ((1.0, 1.0),)},
                        }
            self.ml_calc = GPCalculator(
            kernel_dict=self.kdict, opt_hyperparam=False, scale_data=False,
            scale_optimizer=False,
            calc_uncertainty=True, regularization=1e-5)

        # Settings of the NEB.
        self.neb_method = neb_method
        self.spring = spring
        self.initial_endpoint = is_endpoint[-1]
        self.final_endpoint = fs_endpoint[-1]
        
        # Create images, first interpolation:

        self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                    fs_endpoint=self.final_endpoint,
                                    images_interpolation=None,
                                    n_images=self.n_images,
                                    constraints=self.constraints,
                                    index_constraints=self.ind_mask_constr,
                                    trained_process=None,
                                    ml_calculator=self.ml_calc,
                                    scaling_targets=self.scale_targets,
                                    iteration=self.iter
                                    )

        neb_interpolation = NEB(self.images, k=self.spring)
        
        neb_interpolation.interpolate(method=self.neb_method)

        self.initial_images = copy.deepcopy(self.images)

    def run(self, fmax=0.05, max_iter=100, ml_algo='FIRE',
            plot_neb_paths=False):

        """Executing run will start the optimization process.

        Parameters
        ----------
        ml_algo : string
            Algorithm for the surrogate model:
            'FIRE_ASE' and 'MDMin_ASE'.
        neb_method: string
            Implemented are: 'aseneb', 'improvedtangent' and 'eb'.
            (ASE function).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html#module-ase.neb
        fmax: float
            Convergence criteria. Forces below fmax.
        plot_neb_paths: bool
            If True it stores the predicted NEB path in pdf format for each
            iteration.
            Note: Python package matplotlib is required.

        Returns
        -------
        The NEB results are printed in filename.txt (results.txt, if the
        user don't specify a name).
        A trajectory file in ASE format (filename.traj)
        containing the geometries of the system for each step
        is also generated. A dictionary with a summary of the results can be
        accessed by self.results and it is also printed in 'filename'_data.txt.

        """

        while True:

            # 1) Train Machine Learning process:

            # Check that you are not feeding redundant information.
            count_unique = np.unique(self.list_train, return_counts=True,
                                     axis=0)[1]
            msg = 'Your training list constains 1 or more ' \
            'duplicated elements'
            assert np.any(count_unique) < 2, msg


            process = train_ml_process(list_train=self.list_train,
                                           list_targets=self.list_targets,
                                           list_gradients=self.list_gradients,
                                           index_constraints=self.ind_mask_constr,
                                           ml_calculator=self.ml_calc,
                                           scaling_targets=self.scale_targets)

            trained_process = process['trained_process']
            ml_calc = process['ml_calc']


            # 2) Setup and run ML NEB:



            starting_path = copy.deepcopy(self.initial_images)

            if self.iter > 1 and np.max(uncertainty_path[1:-1]) < 0.025:
                starting_path = self.images

            self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=starting_path,
                                        n_images=self.n_images,
                                        constraints=self.constraints,
                                        index_constraints=self.ind_mask_constr,
                                        trained_process=trained_process,
                                        ml_calculator=ml_calc,
                                        scaling_targets=self.scale_targets,
                                        iteration=self.iter
                                        )

            ml_neb = NEB(self.images, climb=True,
                      method=self.neb_method,
                      k=self.spring)
            neb_opt = eval('MDMin')(ml_neb, dt=0.01)

            neb_opt.run(fmax=fmax,
                        steps=150)

            # 3) Get results from ML NEB:

            # Get fit of the discrete path.
            neb_tools = NEBTools(self.images)
            [s, E, Sfit, Efit, lines] = neb_tools.get_fit()

            uncertainty_path = []
            energies_path = []
            for i in self.images:
                uncertainty_path.append(i.info['uncertainty'])
                energies_path.append(i.get_total_energy())


            # Select image with max. uncertainty.


            if self.iter % 2 == 0:
                argmax_unc = np.argmax(uncertainty_path[1:-1])
                interesting_point = self.images[1:-1][
                                          argmax_unc].get_positions().flatten()


            if self.iter % 2 == 1:
                argmax_unc = np.argmax(np.abs(energies_path[1:-1]))
                interesting_point = self.images[1:-1][
                                          argmax_unc].get_positions().flatten()


            # Store plots.

            if plot_neb_paths is True:
                neb_tools.plot_band()

                plt.errorbar(s[1:-1], E[1:-1], yerr=uncertainty_path[1:-1],
                             ls='none', ecolor='black', capsize=10.0)
                plt.suptitle('Iteration: {0:.0f}'.format(self.iter), x=0.85,
                             y=0.9)
                plt.axvline(x=s[argmax_unc+1])
                plt.show()


            if plot_neb_paths is True:
                if self.ase_calc.__dict__['name'] == 'mullerbrown':
                    plot_neb_mullerbrown(images=self.images,
                                         interesting_point=interesting_point,
                                         trained_process=trained_process,
                                         list_train=self.list_train)

            # 3) Add a new training point and evaluate it.

            evaluate_interesting_point_and_append_training(self,
                                                          interesting_point)

            # 4) Save (append) opt. path (including initial and final images).

            # Store all the atoms objects of the real evaluations (expensive).
            TrajectoryWriter(atoms=self.ase_ini, filename='./' + str(
                             self.filename) +'_evaluated_images.traj',
                             mode='a').write()

            print('Length of initial path:', self.d_start_end)
            print('Length of the current path:', s[-1])
            print('Max uncertainty:', np.max(uncertainty_path))
            print('ITERATIONS:', self.iter)

            # Break if converged:

            max_forces = get_fmax(-np.array([self.list_gradients[-1]]),
                                    self.num_atoms)
            max_abs_forces = np.max(np.abs(max_forces))

            print('Forces last image evaluated:', max_abs_forces)
            print('Energy of the last image evaluated:',
                      self.ase_ini.get_total_energy()-self.scale_targets)
            print('Image number:', argmax_unc+2)

            if max_abs_forces <= fmax:
                print("\n Congratulations your NEB path is converged!")
                print('Image number:', argmax_unc+2)

                break

            # Break if reaches the max number of iterations set by the user.
            if max_iter <= self.iter:
                warning_max_iter_reached()
                break

        # Print Final convergence:
        print('Number of function evaluations in this run:', self.iter)
