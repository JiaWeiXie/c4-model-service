from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.module import MODULE_OPTIMIZERS
from torch import nn
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaModel,
    get_linear_schedule_with_warmup,
)

from c4.model import Seq2Seq
from c4service.utils import init_tokenizer


class C4Model(LightningModule):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        beam_size: int = 10,
        max_length: int = 512,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 3,
        train_data_len: int = 1,
    ) -> None:
        super().__init__()

        if not model_name_or_path:
            self.model_name_or_path = "microsoft/graphcodebert-base"
            # self.model_name_or_path = "microsoft/codebert-base"
        else:
            self.model_name_or_path = model_name_or_path

        self.tokenizer = init_tokenizer(self.model_name_or_path)
        self.config = RobertaConfig.from_pretrained(self.model_name_or_path)
        self.encoder = RobertaModel.from_pretrained(
            self.model_name_or_path,
            config=self.config,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_attention_heads,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=6,
        )
        self.model = Seq2Seq(
            encoder=self.encoder,
            decoder=self.decoder,
            config=self.config,
            beam_size=beam_size,
            max_length=max_length,
            sos_id=self.tokenizer.cls_token_id,
            eos_id=self.tokenizer.sep_token_id,
        )
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.train_data_len = train_data_len

    def load_state_file(
        self,
        load_model_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        path = Path(load_model_path)
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)

        if path.exists():
            self.model.load_state_dict(
                torch.load(
                    path,
                    map_location=device,
                ),
            )

    def forward(
        self,
        source_ids: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        return_single: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": None if return_single else target_ids,
            "target_mask": None if return_single else target_mask,
            "args": None if not return_single else "single",
        }
        return self.model(**inputs)

    def configure_optimizers(self) -> MODULE_OPTIMIZERS:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        total = (
            self.train_data_len
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total * 0.1),
            num_training_steps=total,
        )
        return [optimizer], [scheduler]
