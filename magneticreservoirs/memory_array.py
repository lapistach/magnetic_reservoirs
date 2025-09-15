import numpy as np

class memory_array:
    """
    This class is used to create a memory array that can be used to store and manipulate memory coefficients.
    It provides methods to initialize the memory array and to get the memory coefficients.
    """
    
    def __init__(self, memory_coefficients: np.array = np.array([[1., 0.],[1., 0.45]])):
        self.memory_coefficients = memory_coefficients

    def psw_init(self, default: bool = True, psw: list = [0., 0.]) -> np.array:
        """
        This function initializes the psw (probability of switching) array with the given values.
        If no values are provided, it defaults to an array of zeros.
        """
        if default:
            return np.zeros(self.memory_coefficients.shape[0])
        return np.array(psw)


    def pulses(self, pulse: np.array = np.array([0., 1., 0.]), last_psw: np.array = np.array([0., 0.])) -> list:
        """
        This function takes a pulse and memory coefficients 
        and returns a list of len(memory coefficients) of pulses that are multiplied by 
        (a + b * last_psw[i]) for each couple [a,b][i].
        """
        return [pulse * (self.memory_coefficients[i][0] + self.memory_coefficients[i][1] * last_psw[i]) for i in range(len(self.memory_coefficients))]

