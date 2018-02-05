"""Base class for the feature generators."""
from atoml.api.ase_atoms_api import extend_atoms_class
from atoml.utilities.neighborlist import atoms_neighborlist


class FeatureGenerator(object):
    """Base class for feature generation."""

    def __init__(self, dtype='atoms'):
        """Initialize the class.

        Parameters
        ----------
        dtype : str
            Expected data type. Currently only supports ase atoms objects.
        """
        self.dtype = dtype

    def make_neighborlist(self, candidate, dx=None, neighbor_number=1):
        """Function to generate the neighborlist.

        Parameters
        ----------
        candidate : object
            Target data object on which to generate neighbor list.
        dx : dict
            Buffer to calculate nearest neighbor pairs in dict format:
            dx = {atomic_number: buffer}.
        neighbor_number : int
            Neighbor shell.
        """
        if self.dtype == 'atoms':
            extend_atoms_class(candidate)
            nl = atoms_neighborlist(candidate, dx, neighbor_number)
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

        candidate.set_neighborlist(nl)

        return nl

    def get_neighborlist(self, candidate):
        """Function to return the neighborlist.

        It will check to see if the neighbor list is stored in the data object.
        If not it will generate the neighborlist from scratch.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the neighbor list.
        """
        try:
            nl = candidate.get_neighborlist()
        except AttributeError:
            nl = None

        if nl is None:
            nl = self.make_neighborlist(candidate)

        return nl

    def get_positions(self, candidate):
        """Function to return the atomic coordinates.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic coordinates.
        """
        if self.dtype == 'atoms':
            return candidate.get_positions()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_atomic_numbers(self, candidate):
        """Function to return the atomic numbers.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic numbers.
        """
        if self.dtype == 'atoms':
            return candidate.get_atomic_numbers()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_chemical_symbols(self, candidate):
        """Function to return the atomic symbols.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic symbols.
        """
        if self.dtype == 'atoms':
            return candidate.get_chemical_symbols()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_masses(self, candidate):
        """Function to return the atomic masses.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic masses.
        """
        if self.dtype == 'atoms':
            return candidate.get_masses()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_all_distances(self, candidate):
        """Function to return the atomic distances.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic distances.
        """
        if self.dtype == 'atoms':
            return candidate.get_all_distances()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))
