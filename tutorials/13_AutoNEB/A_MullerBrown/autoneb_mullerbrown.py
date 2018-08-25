from ase.io import read, write
from ase.optimize import BFGS
import copy
from ase.autoneb import AutoNEB
from catlearn.optimize.catlearn_autoneb_optimizer import CatLearnAutoNEB
from catlearn.optimize.functions_calc import MullerBrown
from ase import Atoms
import glob
import shutil
import os


# 0. Clean up before running. ################################################

# Check there are no previous images:
for i in glob.glob("images*"):
    os.remove(i)

# Remove data from previous iteration:
directory = './AutoNEB_iter'
if os.path.exists(directory):
    shutil.rmtree(directory)

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = MullerBrown()

# # 1.1. Structures:
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.626, 0.025, 0.0)])

initial_structure.set_calculator(copy.deepcopy(ase_calculator))
final_structure.set_calculator(copy.deepcopy(ase_calculator))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
initial_opt = BFGS(initial_structure, trajectory='initial.traj')
initial_opt.run(fmax=0.01)

# Final end-point:
final_opt = BFGS(final_structure, trajectory='final.traj')
final_opt.run(fmax=0.01)

# Define number of images:
n_images = 11

# 2.A. AutoNEB using ASE  ####################################################

# Write in format for AutoNEB:
initial = read('initial.traj')
write('images000.traj', initial)

final = read('final.traj')
write('images001.traj', final)


def attach_calculators(images):
    for i in range(len(images)):
        images[i].set_calculator(MullerBrown())


autoneb_ase = AutoNEB(attach_calculators,
                      prefix='images',
                      optimizer='BFGS',
                      n_simul=1,
                      n_max=n_images,
                      fmax=0.05,
                      k=0.5,
                      parallel=False,
                      maxsteps=[50, 1000])

autoneb_ase.run()

# 2.B. AutoNEB using CatLearn ################################################

autoneb_catlearn = CatLearnAutoNEB(start='initial.traj',
                                   end='final.traj',
                                   ase_calc=copy.deepcopy(ase_calculator),
                                   n_images=n_images)
autoneb_catlearn.run(fmax=0.05, plot_neb_paths=True)