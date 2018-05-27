from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS
from ase.visualize import view
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import NEBOptimizer
from ase.neb import NEBTools
import copy
from catlearn.optimize.mullerbrown_calc import MullerBrown
from ase import Atoms
from ase.optimize import LBFGS, FIRE, BFGS, BFGSLineSearch, MDMin
import numpy as np

""" Toy model using the MullerBrown potential.  
In this tutorial, we first optimize the initial and final end-points of the 
reaction path. Secondly, we make a comparison between the ASE 
implementation and our CatLearn (Machine Learning assisted) NEB code.
"""


# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = MullerBrown()

# 1.1. Structures:
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.625, 0.025, 0.0)])

initial_structure.set_calculator(copy.deepcopy(ase_calculator))
final_structure.set_calculator(copy.deepcopy(ase_calculator))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
initial_opt = FIRE(initial_structure, trajectory='initial_optimized.traj')
initial_opt.run(fmax=0.01)

# Final end-point:
final_opt = FIRE(final_structure, trajectory='final_optimized.traj')
final_opt.run(fmax=0.01)

# 2.A. NEB using ASE #########################################################

number_of_images_neb_ase = 9
initial_ase = read('initial_optimized.traj')
final_ase = read('final_optimized.traj')
images_ase = [initial_ase]
for i in range(1, number_of_images_neb_ase-1):
    image_ase = initial_ase.copy()
    image_ase.set_calculator(copy.deepcopy(ase_calculator))
    images_ase.append(image_ase)
images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True, method='improvedtangent', k=0.1)
neb_ase.interpolate()

qn_ase = FIRE(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.01)

nebtools_ase = NEBTools(images_ase)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()
plt.show()


# 2.B. NEB using CatLearn ####################################################

initial = read('initial_optimized.traj')
final = read('final_optimized.traj')

neb_catlearn = NEBOptimizer(start='initial_optimized.traj',
                       end='final_optimized.traj',
                       ase_calc=copy.deepcopy(ase_calculator),
                       n_images=9, interpolation='')

neb_catlearn.run(ml_algo='MDMin', climb_img=True, max_step=0.10,
                 neb_method='improvedtangent', store_neb_paths=True)

# 3. Summary of the results #################################################