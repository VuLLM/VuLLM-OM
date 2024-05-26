
from typing import Optional, List
import torch
from trl import SFTTrainer
from transformers.integrations.deepspeed import deepspeed_init
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize
from transformers.utils import logging
# from torch import xla
# if is_torch_xla_available():
#     import xla.core.xla_model as xm
#     import xla.debug.metrics as met

class Custom_SFTTrainer(SFTTrainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        logger = logging.get_logger(__name__)

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize accumulators for metrics
        total_loss = 0.0
        total_accuracy = 0.0
        total_google_bleu = 0.0
        total_sacre_bleu = 0.0
        total_gen_len = 0
        total_batches = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update accumulators
            if loss is not None:
                total_loss += loss.item()

            if logits is not None and labels is not None:
                # Compute batch-level metrics
                logits.cpu().numpy()
                labels.cpu().numpy()
                batch_metrics = self.compute_metrics((logits, labels))

                total_accuracy += batch_metrics.get('eval_accuracy', 0)
                total_google_bleu += batch_metrics.get('googleBleu', 0)
                total_sacre_bleu += batch_metrics.get('sacreBleu', 0)
                total_gen_len += batch_metrics.get('gen_len', 0)

            total_batches += 1

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # Finalize metrics
        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        average_google_bleu = total_google_bleu / total_batches
        average_sacre_bleu = total_sacre_bleu / total_batches
        average_gen_len = total_gen_len / total_batches

        metrics = {
            f"{metric_key_prefix}_loss": average_loss,
            f"{metric_key_prefix}_accuracy": average_accuracy,
            f"{metric_key_prefix}_googleBleu": average_google_bleu,
            f"{metric_key_prefix}_sacreBleu": average_sacre_bleu,
            f"{metric_key_prefix}_gen_len": average_gen_len,
        }

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=self.num_examples(dataloader) if has_length(dataloader) else None)

