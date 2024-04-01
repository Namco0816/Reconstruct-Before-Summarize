from dataclasses import dataclass
from itertools import accumulate
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
import torch.nn as nn
import torch

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
                
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # TODO or NOT: keep the <s> token, make the feature['anchor_mask'][1] = 1
        anchor_merge_index = [list(accumulate(feature['anchor_mask'])) for feature in features]
        anchor_padding_indices = [max(anchor) + 1 for anchor in anchor_merge_index]
        max_anchor_mask_length = max([len(anchor) for anchor in anchor_merge_index])

        if self.pad_to_multiple_of is not None:
            max_anchor_mask_length = (
               (max_anchor_mask_length + self.pad_to_multiple_of - 1) 
               // self.pad_to_multiple_of 
               * self.pad_to_multiple_of 
            )
        for i, feature in enumerate(features):
            anchor_remainder = [anchor_padding_indices[i]] * (max_anchor_mask_length - len(anchor_merge_index[i]))
            if isinstance(anchor_merge_index[i], list):
                feature["anchor_merge_index"] = anchor_merge_index[i] + anchor_remainder
            else:
                feature["anchor_merge_index"] = np.concatenate([anchor_merge_index[i], anchor_remainder]).astype(np.int64)

            del feature['anchor_mask']

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        anchor_merge_index = torch.tensor(features['anchor_merge_index']).long()
        batch_size, _ = anchor_merge_index.shape
        max_src_length = torch.max(anchor_merge_index) + 1
        anchor_merge_index = anchor_merge_index.unsqueeze(2)
        embed_device = self.model.get_encoder().embed_tokens.weight.device

        orig_input_embeds = self.model.get_encoder().embed_tokens(features['input_ids'].to(embed_device))

        anchor_merge_index = anchor_merge_index.expand(-1, -1, orig_input_embeds.shape[-1]).to(embed_device)
        inputs_embeds = torch.zeros(batch_size, max_src_length, orig_input_embeds.shape[-1]).to(embed_device)

        inputs_embeds = inputs_embeds.scatter_reduce(
            dim = 1, 
            index = anchor_merge_index, 
            src = orig_input_embeds, 
            reduce = 'mean',
            include_self = False
            )

        features['inputs_embeds'] = inputs_embeds.cpu()
        batch_size, seq_len, dim = inputs_embeds.shape
        features['attention_mask'] = features['attention_mask'][:, :seq_len]
        del features['input_ids']
        del features['anchor_merge_index']

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class DataCollatorForSoftTruncation:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
                
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # TODO or NOT: keep the <s> token, make the feature['anchor_mask'][1] = 1
        anchor_merge_index = [list(accumulate(feature['anchor_mask'])) for feature in features]
        anchor_padding_indices = [max(anchor) + 1 for anchor in anchor_merge_index]
        max_anchor_mask_length = max([len(anchor) for anchor in anchor_merge_index])

        if self.pad_to_multiple_of is not None:
            max_anchor_mask_length = (
               (max_anchor_mask_length + self.pad_to_multiple_of - 1) 
               // self.pad_to_multiple_of 
               * self.pad_to_multiple_of 
            )
        for i, feature in enumerate(features):
            anchor_remainder = [anchor_padding_indices[i]] * (max_anchor_mask_length - len(anchor_merge_index[i]))
            if isinstance(anchor_merge_index[i], list):
                feature["anchor_merge_index"] = anchor_merge_index[i] + anchor_remainder
            else:
                feature["anchor_merge_index"] = np.concatenate([anchor_merge_index[i], remainder]).astype(np.int64)

            del feature['anchor_mask']

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features['anchor_merge_index'] = features['anchor_merge_index'].long()        

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class DataCollatorForImportanceScatter:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
                
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        anchor_merge_index = [feature['buckets_order'] for feature in features]

        anchor_padding_indices = [max(anchor) + 1 for anchor in anchor_merge_index]
        max_anchor_mask_length = max([len(anchor) for anchor in anchor_merge_index])

        if self.pad_to_multiple_of is not None:
            max_anchor_mask_length = (
               (max_anchor_mask_length + self.pad_to_multiple_of - 1) 
               // self.pad_to_multiple_of 
               * self.pad_to_multiple_of 
            )
        for i, feature in enumerate(features):
            anchor_remainder = [anchor_padding_indices[i]] * (max_anchor_mask_length - len(anchor_merge_index[i]))

            if anchor_remainder != []:
                feature['attention_mask'] = (max(anchor_merge_index[i]) + 1) * [1] + [0]
            else:
                feature['attention_mask'] = (max(anchor_merge_index[i]) + 1) * [1]

            if isinstance(anchor_merge_index[i], list):
                feature["anchor_merge_index"] = anchor_merge_index[i] + anchor_remainder
            else:
                feature["anchor_merge_index"] = np.concatenate([anchor_merge_index[i], anchor_remainder]).astype(np.int64)

            del feature['buckets_order']

        attn_masks = [feature['attention_mask'] for feature in features]
        max_attn_len = max([len(l) for l in attn_masks])
        
        if self.pad_to_multiple_of is not None:
            max_attn_len = (
                (max_attn_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        for i, feature in enumerate(features):
            attn_remainder = [0] * (max_attn_len - len(feature['attention_mask']))
            if isinstance(feature['attention_mask'], list):
                feature['attention_mask'] = feature['attention_mask'] + attn_remainder
            else:
                feature['attention_mask'] = np.concatenate([feature['attention_mask'], attn_remainder]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features['anchor_merge_index'] = features['anchor_merge_index'].long()        
        assert features['anchor_merge_index'].shape == features['input_ids'].shape

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features