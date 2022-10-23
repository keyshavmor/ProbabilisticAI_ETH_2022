import os
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gpytorch
from matplotlib import rcParams
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ExactGPModel

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0

class Model(object):
    def __init__(self):
        self.rng = np.random.default_rng(seed=0)

        self.model = None
        self.likelihood = None
            
    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        test_x_tensor = torch.Tensor(test_features)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_of_x = self.model(test_x_tensor)
            mean = f_of_x.mean.cpu().numpy()
            _, f_of_x_upper = f_of_x.confidence_region()
            f_of_x_upper = f_of_x_upper.cpu().numpy()
            y_star = self.likelihood(f_of_x)
            y_star_lower, y_star_upper = y_star.confidence_region()
            
        return mean, mean, f_of_x_upper
        
    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        
        train_X, val_X, train_Y, val_Y = train_test_split(train_features,train_GT, train_size=0.99,test_size=0.01,random_state=0,shuffle=True)
        val_X = torch.Tensor(val_X)
        val_Y = torch.Tensor(val_Y).squeeze()
        train_tensor_x = torch.Tensor(train_X)
        train_tensor_y = torch.Tensor(train_Y).squeeze()
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_tensor_x,train_tensor_y,likelihood)
        #optimizer_1 = torch.optim.Adam(model.parameters(), lr=2)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=23,history_size=10)
        #optimizer = torch.optim.Rprop(model.parameters(), lr=1.0, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        scheduler_reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, min_lr=0.05, threshold=0.001,eps=1e-08, verbose=True)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)
        model.train()
        likelihood.train()
        # Fit the model
        print('Fitting model')
        global last_best_loss
        
        def closure():
        # Zero gradients from previous iteration
        #for iteration in range(30):
            optimizer.zero_grad()
            # Output from model
            output = model(train_tensor_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_tensor_y)
            last_best_loss = loss
            loss.backward()
            print('Loss: %.3f, LR: %.3f' , (loss.item(),optimizer.param_groups[0]["lr"]))
            if(loss.item() < 10):
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=9999999999.0)
            scheduler_reduce_lr.step(loss)
            return loss
        optimizer.step(closure)
        self.model = model
        self.likelihood = likelihood


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
