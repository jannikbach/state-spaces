import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


class PlotPredictionVersusGroundTruth(Callback):
    """Plot the values of the prediction and the ground truth in the test phase"""

    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        print("we are at test epoch end")
        predicted_output = []
        ground_truth = []
        for batch_output in trainer.callback_metrics['test_loss']:
            predicted_output.append(batch_output['y_hat'].detach().cpu().numpy())
            ground_truth.append(batch_output['y'].detach().cpu().numpy())

        # Concatenate predicted output and ground truth across batches
        predicted_output = np.concatenate(predicted_output, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        # Create a plot that displays both the predicted values and the ground truth values for the feature
        plt.plot(predicted_output, label='Predicted')
        plt.plot(ground_truth, label='Ground Truth')
        plt.legend()
        plt.title('Predicted vs. Ground Truth')
        plt.xlabel('Time')
        plt.ylabel('Feature Value')
        plt.show()

        # Log the plot to WandB
        trainer.logger.experiment.log({'Predicted vs. Ground Truth': plt})
