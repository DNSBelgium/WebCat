from typing import Optional
from typing import Tuple

import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaForSequenceClassification
from transformers import logging
from transformers.modeling_outputs import SequenceClassifierOutput

from config import PRETRAINED_MODEL, NB_EXTRA_FEATURES, DEVICE


class CustomXlmRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_extra_dims):
        super().__init__()
        total_dims = config.hidden_size + num_extra_dims
        self.dense = nn.Linear(total_dims, total_dims)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(total_dims, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomXlmRoberta(XLMRobertaForSequenceClassification):
    """
    An XLM-RoBERTa model for sequence classification, but modified to incorporate other (numerical) features. These
    features are appended to the XLM-R outputs before those are passed to the classification head.
    """
    def __init__(self, config, num_extra_dims):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = CustomXlmRobertaClassificationHead(config, num_extra_dims)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            extra_data: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor] | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output will be (batch_size, seq_length, hidden_size)
        sequence_output = outputs[0]

        # additional data should be (batch_size, num_extra_dims)
        cls_embedding = sequence_output[:, 0, :]

        if extra_data is not None:
            output = torch.cat((cls_embedding, extra_data), dim=-1)
        else:
            output = cls_embedding

        logits = self.classifier(output)

        loss: None | nn.CrossEntropyLoss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_bert(label_amount: int, move: bool = True) -> CustomXlmRoberta:
    """
    Instantiates a modified XLM-R model from the PRETRAINED_MODEL specified in config.py.

    :param label_amount: Number of output labels/categories.
    :param move: Should the model immediately be moved to the configured DEVICE (specified in config.py).
    """
    saved_verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    model = CustomXlmRoberta.from_pretrained(PRETRAINED_MODEL, num_labels=label_amount, num_extra_dims=NB_EXTRA_FEATURES)
    logging.set_verbosity(saved_verbosity)

    if move:
        model = model.to(DEVICE)

    return model

