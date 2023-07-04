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
        self.data = {
            'train': {'predictions': [], 'ground_truths': [], 'contexts': []},
            'validation': {'predictions': [], 'ground_truths': [], 'contexts': []},
            'test': {'predictions': [], 'ground_truths': [], 'contexts': []},
        }

    def save_step(self, key, outputs, replace=False):
        # if outputs["prediction"].shape[2] > 1:
        #     y_hat = outputs["prediction"][:, :, 0].cpu()
        #     y = outputs["ground_truth"][:, :, 0].cpu()
        #     x = outputs["context"][:, :, 0].cpu()
        # else:
        y_hat = outputs["prediction"].cpu()
        y = outputs["ground_truth"].cpu()
        x = outputs["context"].cpu()

        if replace:
            self.data[key]['predictions'] = [y_hat]
            self.data[key]['ground_truths'] = [y]
            self.data[key]['contexts'] = [x]
        else:
            self.data[key]['predictions'].append(y_hat)
            self.data[key]['ground_truths'].append(y)
            self.data[key]['contexts'].append(x)

    def visualize_and_log(self, key, context_length, prediction_length, plot_count, logger=None):

        # Concatenate predicted output and ground truth across batches
        predicted_output = np.concatenate(self.data[key]['predictions'], axis=0)
        ground_truth = np.concatenate(self.data[key]['ground_truths'], axis=0)
        context = np.concatenate(self.data[key]['contexts'], axis=0)[:, :context_length] # check dim for traffic set.. this should be 2 dimesnional not 3

        # Create a plot that displays both the predicted values and the ground truth values for the feature

        #specific for robot data when obs is not part of target
        dim = 1
        for i in np.random.randint(0, predicted_output.shape[0], plot_count):
            # Create a figure and axis object
            fig, ax = plt.subplots()

            # Plot the first array on the left side
            ax.plot(range(context_length), context[i, :context_length, dim], label='Context')

            # Plot the other two arrays on the right side
            begin_double_plot = (context_length - 1)
            end_double_plot = (context_length + prediction_length)
            ax.plot(range(begin_double_plot, end_double_plot),
                    np.concatenate((context[i, :, dim], predicted_output[i, :, dim]))[begin_double_plot:end_double_plot],
                    label='Predicted')
            ax.plot(range(begin_double_plot, end_double_plot),
                    np.concatenate((context[i, :, dim], ground_truth[i, :, dim]))[begin_double_plot:end_double_plot],
                    label='Ground Truth')

            # Set the x-axis range to be 0 to 150
            ax.set_xlim(0, end_double_plot)

            # Add a legend
            ax.legend()

            # Title and Axis
            plt.title('[' + key + ']' + 'Predicted vs. Ground Truth: ' + str(i))
            plt.xlabel('Time')
            plt.ylabel('Feature Value')

            # Log the plot to WandB
            if logger is not None:
                logger.experiment.log(
                    {'[' + key + ']' + 'Predicted vs. Ground Truth: ' + str(i): plt})  # wandb.Image(plt)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.save_step(key='test', outputs=outputs)


    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.save_step(key='train', outputs=outputs, replace=True)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.save_step(key='validation', outputs=outputs, replace=True)


    @rank_zero_only
    def on_test_end(self, trainer, pl_module):

        context_length = pl_module.task.dataset.dataset_test.context_length
        prediction_length = pl_module.task.dataset.dataset_test.l_output

        self.visualize_and_log(key='test',
                               context_length=context_length,
                               prediction_length=prediction_length,
                               plot_count=5,
                               logger=trainer.logger,
                               )

        self.visualize_and_log(key='validation',
                               context_length=context_length,
                               prediction_length=prediction_length,
                               plot_count=5,
                               logger=trainer.logger,
                               )

        self.visualize_and_log(key='train',
                               context_length=context_length,
                               prediction_length=prediction_length,
                               plot_count=5,
                               logger=trainer.logger,
                               )
