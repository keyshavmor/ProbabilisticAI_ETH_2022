import numpy as np
import warnings
from scipy.optimize import NonlinearConstraint, fmin_l_bfgs_b, minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import GPy


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0
# POI: probability of improvement
# EI: expected improvement
# UCB: GP upper cofidence bound
ACQUISITION_FN = "UCB"

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.f_kernel = GPy.kern.Matern52(input_dim=1, variance=0.5, lengthscale=0.5)
        self.f_mean = 0.0
        self.v_kernel = GPy.kern.Matern52(input_dim=1, variance=1.414, lengthscale=0.5)
        self.v_mean = 1.5
        self.v_min = 1.2

        self.xs = np.zeros((0, domain.shape[0]))
        self.fs = np.zeros((0, 1))
        self.vs = np.zeros((0, 1))

        self.current_f = None
        self.current_v = None
        self.x_best = None
        self.cum_regret = 0.0

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        assert len(self.xs) == len(self.fs) == len(self.vs) and len(self.xs) > 0

        self.f = GPy.models.GPRegression(np.atleast_2d(self.xs), np.atleast_2d(self.fs), self.f_kernel)

        self.v = GPy.models.GPRegression(np.atleast_2d(self.xs), np.atleast_2d(self.vs), self.v_kernel)

        fn_mean = self.f.predict(np.atleast_2d(self.xs))[0]
        xn_best_idx = np.argmax(fn_mean)
        self.x_best = self.xs[xn_best_idx]

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.

        x_opt = self.optimize_acquisition_function()

        return x_opt

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.
    
        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
    
        def objective(x):
            return -self.acquisition_function(x)
       
        def get_constraints():
            assert self.v is not None
            constraint_fn = lambda x: self.v.predict(np.atleast_2d(x))[0][0, 0]
            return NonlinearConstraint(constraint_fn, lb=SAFETY_THRESHOLD, ub=np.inf)
            

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = minimize(objective, x0=x0, method="SLSQP", bounds=domain,
                constraints=get_constraints(), options={"disp": False,
                "finite_diff_rel_step": "3-point"})
            x_values.append(np.clip(result.x, *domain[0]))
            f_values.append(-result.fun)

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        assert self.f is not None
        assert self.x_best is not None

        if ACQUISITION_FN == "POI":
            f_x_best = self.f.predict(np.atleast_2d(self.x_best))[0]
            f_mean, f_var = self.f.predict(np.atleast_2d(x))
            f_std = np.sqrt(f_var[0,0])
            xi = 0.01
            return norm.cdf((f_mean[0, 0] - f_x_best[0, 0] - xi) / f_std)
        elif ACQUISITION_FN == "EI":
            f_x_best = self.f.predict(np.atleast_2d(self.x_best))[0]
            f_mean, f_var = self.f.predict(np.atleast_2d(x))
            f_std = np.sqrt(f_var[0,0])
            xi = 0.01
            imp = (f_mean[0,0] - f_x_best[0,0] - xi)
            if(f_std > 0):
                Z = imp / f_std
                ei = (imp * norm.cdf(Z) + f_std * norm.logpdf(Z))
            else:
                Z = 0
                ei = 0
            return ei
        elif ACQUISITION_FN == "UCB":
            f_mean, f_var = self.f.predict(np.atleast_2d(x))
            f_std = np.sqrt(f_var)
            beta= 2.0
            return (f_mean[0,0] + beta*f_std[0,0])


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.xs = np.vstack((self.xs, x))
        self.fs = np.vstack((self.fs, f))
        self.vs = np.vstack((self.vs, v))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        fn_mean = self.f.predict(np.atleast_2d(self.xs))[0]
        xn_best_idx = np.argmax(fn_mean)
        self.x_best = self.xs[xn_best_idx]

        return self.x_best


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()