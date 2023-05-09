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
    def on_test_epoch_end(self, trainer, pl_module):

        # Concatenate predicted output and ground truth across batches
        predicted_output = np.concatenate(self.predictions, axis=0)
        ground_truth = np.concatenate(self.ground_truths, axis=0)
        context = np.concatenate(self.contexts, axis=0)

        # Create a plot that displays both the predicted values and the ground truth values for the feature
        # todo: remove magic numbers
        for i in np.random.randint(0, predicted_output.shape[0], 20): #range(predicted_output.shape[0]):
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Plot the first array on the left side
            ax.plot(range(75), context[i, :, 0], label='Context')

            # Plot the other two arrays on the right side
            ax.plot(range(74, 150), np.concatenate((context[i, :, 0], predicted_output[i, :, 0]))[74:150], label='Predicted')
            ax.plot(range(74, 150), np.concatenate((context[i, :, 0], ground_truth[i, :, 0]))[74:150], label='Ground Truth')

            # Set the x-axis range to be 0 to 150
            ax.set_xlim(0, 150)

            # Add a legend
            ax.legend()

            # plt.plot(predicted_output[i, :, 0], label='Predicted')
            # plt.plot(ground_truth[i, :, 0], label='Ground Truth')
            # plt.legend()
            plt.title('Predicted vs. Ground Truth: ' + str(i))
            plt.xlabel('Time')
            plt.ylabel('Feature Value')
            # plt.show()

            # Log the plot to WandB
            trainer.logger.experiment.log({'Predicted vs. Ground Truth ' + str(i): plt})  # wandb.Image(plt)



        # plt.plot(predicted_output[0, :, 0], label='Predicted')
        # plt.plot(ground_truth[0, :, 0], label='Ground Truth')
        # plt.legend()
        # plt.title('Predicted vs. Ground Truth')
        # plt.xlabel('Time')
        # plt.ylabel('Feature Value')
        # # plt.show()
        #
        # # Log the plot to WandB
        # trainer.logger.experiment.log({'Predicted vs. Ground Truth': plt}) #wandb.Image(plt)
