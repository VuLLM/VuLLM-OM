
from typing import Dict, Optional, List, Union, Any
import torch
from trl import SFTTrainer
from datasets import Dataset
from transformers.integrations.tpu import tpu_spmd_dataloader
import time
from transformers.trainer_utils import speed_metrics
import math
from transformers.debug_utils import DebugOption
from transformers.utils import is_torch_xla_available
# from torch import xla
# if is_torch_xla_available():
#     import xla.core.xla_model as xm
#     import xla.debug.metrics as met

class Custom_SFTTrainer(SFTTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        with torch.no_grad():
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if isinstance(eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, _eval_dataset in eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=_eval_dataset,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
                return metrics

            # # memory metrics - must set up as early as possible
            # self._memory_tracker.start()

            # eval_dataloader = self.get_eval_dataloader(eval_dataset)
            # if self.is_fsdp_xla_v2_enabled:
            #     eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

            # start_time = time.time()

            # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            # output = eval_loop(
            #     eval_dataloader,
            #     description="Evaluation",
            #     # No point gathering the predictions if there are no metrics, otherwise we defer to
            #     # self.args.prediction_loss_only
            #     prediction_loss_only=True if self.compute_metrics is None else None,
            #     ignore_keys=ignore_keys,
            #     metric_key_prefix=metric_key_prefix,
            # )

            # total_batch_size = self.args.eval_batch_size * self.args.world_size
            # if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            #     start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
            # output.metrics.update(
            #     speed_metrics(
            #         metric_key_prefix,
            #         start_time,
            #         num_samples=output.num_samples,
            #         num_steps=math.ceil(output.num_samples / total_batch_size),
            #     )
            # )

            # self.log(output.metrics)

            # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            #     xm.master_print(met.metrics_report())

            # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

            # self._memory_tracker.stop_and_update_metrics(output.metrics)

            # return output.metrics
