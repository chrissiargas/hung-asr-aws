from typing import Union, Any, Optional

import torch
from torch import nn
from transformers import Seq2SeqTrainer
from typing import Dict
import torch.distributed as dist

class MultiLossTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super(MultiLossTrainer, self).__init__(*args, **kwargs)
        self.enable_losses(self.model.ctc, self.model.predict_duration, self.model.audio_forecasting)
        self.register_custom_metrics()

    def enable_losses(self, ctc_loss: bool, duration_loss: bool, audio_loss: bool):
        self.ctc_loss = ctc_loss
        self.audio_loss = audio_loss
        self.duration_loss = duration_loss

    def register_custom_metrics(self):
        device = self.args.device
        self.loss_tracker = {
            'total_loss': torch.tensor(0.0, device=device),
            'loss_lm': torch.tensor(0.0, device=device),
            'loss_ctc': torch.tensor(0.0, device=device),
            'loss_audio': torch.tensor(0.0, device=device),
            'loss_duration': torch.tensor(0.0, device=device),
            'steps': torch.tensor(0.0, device=device)
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        kwargs = {"num_items_in_batch": num_items_in_batch} if num_items_in_batch is not None else {}
        outputs = model(**inputs, **kwargs)
        loss = outputs.loss

        if self.args.should_log:
            self.loss_tracker['total_loss'] += loss.detach().item()
            self.loss_tracker['loss_lm'] += outputs.loss_lm.item()
            self.loss_tracker['loss_ctc'] += outputs.loss_ctc.item()
            self.loss_tracker['loss_audio'] += outputs.loss_audio.item()
            self.loss_tracker['loss_duration'] += outputs.loss_duration.item()
            self.loss_tracker['steps'] += 1.0

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self.loss_tracker['steps'].item() > 0:
            steps = self.loss_tracker['steps'].item()

            logs['loss/total_loss'] = self.loss_tracker['total_loss'].item() / steps

            logs['loss/lm_loss'] = self.loss_tracker['loss_lm'].item() / steps
            if self.ctc_loss:
                logs['loss/ctc_loss'] = self.loss_tracker['loss_ctc'].item() / steps
            if self.audio_loss:
                logs['loss/audio_loss'] = self.loss_tracker['loss_audio'].item() / steps
            if self.duration_loss:
                logs['loss/duration_loss'] = self.loss_tracker['loss_duration'].item() / steps

            self.register_custom_metrics()

        super().log(logs, *args, **kwargs)