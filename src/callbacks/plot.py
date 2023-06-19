from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import wandb

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PlotPredictionVersusGroundTruth(Callback):
    """Plot the values of the prediction and the ground truth in the test phase"""

    def __init__(self):
        super().__init__()
        self.predictions = []
        self.ground_truths = []
        self.contexts = []

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        y_hat = outputs["prediction"]
        y = outputs["ground_truth"]
        x = outputs["context"]

        self.predictions.append(y_hat)
        self.ground_truths.append(y)
        self.contexts.append(x)



    @rank_zero_only
    def on_test_end(self, trainer, pl_module):

        context_length = pl_module.task.dataset.dataset_test.context_length
        prediction_length = pl_module.task.dataset.dataset_test.prediction_length

        # Concatenate predicted output and ground truth across batches
        predicted_output = np.concatenate(self.predictions, axis=0)
        ground_truth = np.concatenate(self.ground_truths, axis=0)

        context = np.concatenate(self.contexts, axis=0)[:, :context_length, :]
        # Create a plot that displays both the predicted values and the ground truth values for the feature
        for i in np.random.randint(0, predicted_output.shape[0], 20): #range(predicted_output.shape[0]):
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Plot the first array on the left side
            ax.plot(range(context_length), context[i, :context_length, 0], label='Context')

            # Plot the other two arrays on the right side
            begin_double_plot = (context_length - 1)
            end_double_plot = (context_length + prediction_length)
            ax.plot(range(begin_double_plot, end_double_plot), np.concatenate((context[i, :, 0], predicted_output[i, :, 0]))[begin_double_plot:end_double_plot], label='Predicted')
            ax.plot(range(begin_double_plot, end_double_plot), np.concatenate((context[i, :, 0], ground_truth[i, :, 0]))[begin_double_plot:end_double_plot], label='Ground Truth')

            # Set the x-axis range to be 0 to 150
            ax.set_xlim(0, end_double_plot)

            # Add a legend
            ax.legend()

            # Title and Axis
            plt.title('Predicted vs. Ground Truth: ' + str(i))
            plt.xlabel('Time')
            plt.ylabel('Feature Value')

            # Log the plot to WandB
            if trainer.logger is not None:
                trainer.logger.experiment.log({'Predicted vs. Ground Truth: ' + str(i): plt})  # wandb.Image(plt)

