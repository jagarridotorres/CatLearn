from ase.build import fcc100, add_adsorbate
from ase.calculators.emt import EMT
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
import matplotlib.pyplot as plt
from catlearn.optimize.catlearn_neb_optimizer import NEBOptimizer
from ase.neb import NEBTools
import copy

# 1. Structural relaxation. ##################################################

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(copy.deepcopy(ase_calculator))

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

slab[12].magmom = 2.0

# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = FIRE(slab, trajectory='initial.traj')
qn.run(fmax=0.01)

slab[12].magmom = 1.0

# Final end-point:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = FIRE(slab, trajectory='final.traj')
qn.run(fmax=0.01)


# # Define number of images:
n_images = 9

# 2.A. NEB using ASE #########################################################

initial_ase = read('initial.traj')
final_ase = read('final.traj')
constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial_ase])


images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(copy.deepcopy(ase_calculator))
    image.set_constraint(constraint)
    images_ase.append(image)

images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True, method='improvedtangent', k=0.1)
neb_ase.interpolate(method='idpp')
qn_ase = FIRE(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)
nebtools_ase = NEBTools(images_ase)

Sf_ase = nebtools_ase.get_fit()[2]
Ef_ase = nebtools_ase.get_fit()[3]

Ef_neb_ase, dE_neb_ase = nebtools_ase.get_barrier(fit=False)
nebtools_ase.plot_band()

plt.show()

# 2.B. NEB using CatLearn ####################################################

neb_catlearn = NEBOptimizer(start='initial.traj', end='final.traj',
                            ase_calc=copy.deepcopy(ase_calculator),
                            n_images=n_images,
                            interpolation='idpp')

neb_catlearn.run(fmax=0.05, ml_algo='FIRE', plot_neb_paths=True)


# 3. Summary of the results #################################################

# NEB ASE:
print('\nSummary of the results:\n')

atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = len(atoms_ase) - 2 * n_images

print('Number of function evaluations CI-NEB implemented in ASE:', n_eval_ase)

# Catlearn:
atoms_catlearn = read('evaluated_structures.traj', ':')
n_eval_catlearn = len(atoms_catlearn)
print('Number of function evaluations Catlearn ASE:', n_eval_catlearn-2)
