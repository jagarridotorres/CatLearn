from catlearn.optimize.io import *


def warning_using_ase():
    msg = "Atoms have been detected. Using ASE..."
    print(msg)
    store_warnings_and_errors(msg)


def warning_first_step_ase(self):
    msg = "WARNING: To train a ML algorithm it is required to evaluate at " \
        "least two points. You have chosen to calculate these two initial " \
        "points of this optimization " \
        "using " + str(self.i_ase_step) + " as implemented in ASE. If the " \
        "user don't set the 'i_ase_step' flag the default algorithm is " \
        "ASE_BFGS for atomistic systems.)"
    print(msg)
    store_warnings_and_errors(msg)


def warning_prev_data_intro():
    msg = """WARNING: User is introducing previously evaluated data.
            Remember that the data must be passed as a dictionary 'train', 
            'targets' (and 'gradients', optional)."""
    print(msg)
    store_warnings_and_errors(msg)


def warning_ml_algo(self):
    msg = 'Using ASE (' + self.ml_algo + ') for the optimization of ' \
                'the ML predicted function.'
    print(msg)
    store_warnings_and_errors(msg)


def warning_traj_detected():
    msg = 'Trajectory file has been detected. Using ASE...'
    print(msg)
    store_warnings_and_errors(msg)


def warning_max_iter_reached():
    msg = 'WARNING: Maximum number iterations reached. Not converged.'
    print(msg)
    store_warnings_and_errors(msg)


def warning_max_step():
    msg = "WARNING: You have not set a maximum step size. Notice that " \
        "this can lead to instability problems in the optimization."
    ""
    print(msg)
    store_warnings_and_errors(msg)


def warning_max_step_radii(max_step):
    msg = "WARNING: You have not set a minimum step size and is using " \
        "Atoms objects. In order to stabilize the optimization the default " \
        "maximum step size is 1/4 of the smallest covalent radii " \
        "found for the system. For this optimization the maximum step is set " \
        "to " + \
        str(max_step)+" Angstrom."
    print(msg)
    store_warnings_and_errors(msg)


def warning_max_step_radii_neb(max_step):
    msg = "WARNING: You have not set a minimum step size and is using " \
        "Atoms objects. In order to stabilize the optimization the default " \
        "maximum step size is 1/10 of the smallest covalent radii " \
        "found for the system. For this optimization the maximum step is set " \
        "to " + \
        str(max_step)+" Angstrom."
    print(msg)
    store_warnings_and_errors(msg)

def warning_spring_default(spring):
    msg = "WARNING: You have not set an spring constant. The default spring " \
          "constant for this optimization is k=" + str(spring)+" Angstrom^2."
    print(msg)
    store_warnings_and_errors(msg)


def warning_kernel():
    msg = "WARNING: You have not selected a kernel (covariance " \
            "function) for the GP. Default: Squared Exponential kernel (" \
            "gaussian). This kernels is meant to be used for " \
            "smooth functions (defaults are designed for best performance " \
            "for atomistic systems)."
    print(msg)
    store_warnings_and_errors(msg)


def warning_climb_img(self):
    if self.ci is False:
        msg = "The predicted mean is optimized using NEB (CI-NEB is " \
        "switched off)."""
    if self.ci is True:
        msg = "The predicted mean is optimized using CI-NEB."""
    print(msg)
    store_warnings_and_errors(msg)


def err_first_step_ase():
    msg = """ERROR: In order to work with ASE you must pass an atoms object 
            or trajectory file. Stopping optimization.
            \n HINT: Create an Atoms object using ASE and pass it in x0.
            \n HINT_2: Remember that you can also pass a trajectory file from 
            a previous calculation."""
    store_warnings_and_errors(msg)
    return msg


def err_not_ase_algoml():
    msg = """COMPUTER SAYS NO: You are trying to optimize the ML 
    predicted mean 
    (or 
    acquisition function) using an ASE minimizer with features that are 
    not Atoms objects. 
    \n HINT: Please use a Scipy algorithms instead of an ASE minimizer. 
    Implemented Scipy algorithms are: 'BFGS', 'L-BFGS-B', 'CG', 'Nelder-Mead' 
    and 'Powell'. """
    store_warnings_and_errors(msg)
    return msg


def err_not_x0():
    msg = """COMPUTER SAYS NO: User has not provided an starting point for the
            "optimization. The starting point can be an Atoms or trajectory 
            object (ASE based). Alternatively, you can provide an 
            array with the starting point or a dictionary containing list of 
            'train', 'targets' (and gradients', optional).
            """
    store_warnings_and_errors(msg)
    return msg


def err_not_real_func(self):
    msg = "COMPUTER SAYS NO: An objective function must be introduced.\n"
    msg += "HINT: Select one of the functions implemented in functions.py.\n"
    msg += "Implemented functions are:" + str(self.fun_avail) + "\n"
    msg += "Alternatively, you can add a function into functions.py"
    msg += " In this case, the analytical function and its first derivative " \
        "in each dimension must be added by you. "
    store_warnings_and_errors(msg)
    return msg


def err_not_real_func_2(self):
    msg = "COMPUTER SAYS NO: The selected function is not in the list of " \
    "available " \
        "functions\n."
    msg += "HINT: Select one of the functions implemented in functions.py.\n"
    msg += "Implemented functions are:" + str(self.fun_avail) + "\n"
    msg += "Alternatively, you can add a function into functions.py"
    msg += " In this case, the analytical function and its first derivative " \
        "in each dimension must be added by manually the user. "
    store_warnings_and_errors(msg)
    return msg


def err_train_data():
    msg = "COMPUTER SAYS NO: You must include at least two trained points."
    store_warnings_and_errors(msg)
    return msg


def err_target_data():
    msg = "COMPUTER SAYS NO: You must include at least two target features."
    store_warnings_and_errors(msg)
    return msg


def err_gradients_data():
    msg = "COMPUTER SAYS NO: You must include at least two gradients."
    store_warnings_and_errors(msg)
    return msg


def err_diff_data_size():
    msg = """COMPUTER SAYS NO: You have introduced a different number of 
    train, target and 
                gradients. Please be consistent!"""
    store_warnings_and_errors(msg)
    return msg

def err_not_ase_calc_atoms():
    msg = "COMPUTER SAYS NO: The Atoms object that you passed has not " \
          "calculator attached and you have not passed any calculator the " \
          "'ase calc' flag. You can attach the calculator to the atoms " \
          "object prior feeding it to CatLearn or alternatively you can pass it " \
          "to CatLearn in the 'ase_calc' flag."
    store_warnings_and_errors(msg)
    return msg

def err_not_ase_calc_traj():
    msg = "COMPUTER SAYS NO: When importing a trajectory file you must " \
    "include the ASE calculator ('ase_calc' flag). Please be " \
          "consistent with your previous run."
    store_warnings_and_errors(msg)
    return msg


def err_not_ml_algo(self):
    msg = "COMPUTER SAYS NO: Please set a ML optimizer. Implemented Scipy " \
                "optimizers:" + str(self.scipy_optimizers) + ". Implemented " \
                "ASE " \
                "optimizers (" \
                "only " \
                "for input structures in ASE format, i.e. Atoms objects or " \
                "trajectory files): " + str(self.ase_optimizers) + "."
    store_warnings_and_errors(msg)
    return msg


def err_not_neb_start():
    msg = """COMPUTER SAYS NO: User has not provided an initial structure 
            for the NEB path. The starting point must be an Atoms object or 
            a trajectory file (ASE). If passing an Atoms object please make 
            sure  that your the structure you pass is converged. If passing a 
            trajectory file, the last image must contain the converged 
            structure).
            """
    store_warnings_and_errors(msg)
    return msg


def err_not_neb_end():
    msg = """COMPUTER SAYS NO: User has not provided a final structure 
            for the NEB path. The final point must be an Atoms object or 
            a trajectory file (ASE). If passing an Atoms object please make 
            sure that your the structure you pass is converged. If passing a 
            trajectory file, the last image must contain the converged 
            structure).
            """
    store_warnings_and_errors(msg)
    return msg

def err_not_enough_images():
    msg = """ COMPUTER SAYS NO: You must include at least 4 NEB images."""
    store_warnings_and_errors(msg)
    return msg

def store_warnings_and_errors(msg):
    warn_and_errors = open('warnings_and_errors.txt', "a")
    warn_and_errors.write(msg + "\n")
    warn_and_errors.close()