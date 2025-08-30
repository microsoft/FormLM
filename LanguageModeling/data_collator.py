import random
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Tuple

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch


@dataclass
class DataCollatorForFormLMContinualPretrain(DataCollatorMixin):
    """
    Data collator used for FormLM continual pre-training.

    We continually pre-train FormLM with two structure-aware objectives, Span Masked Language Model (SpanMLM) and Block
    Type Permutation (BTP).

    Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.
    For sequence-to-sequence model, use the uncorrupted input as the decoder_input_ids.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    mlm: bool = False
    span_mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        # We use <sep>, "|" and special tokens to separate properties of a form.
        self.block_sep_token_ids = list(self.tokenizer.get_added_vocab().values())
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids("<sep>")
        self.option_sep_token_id = self.tokenizer.convert_tokens_to_string("|")

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError('Tensorflow framework is not supported yet!')

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # Prepare decoder_input_ids
        if self.model is not None:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["input_ids"])
            batch["decoder_input_ids"] = decoder_input_ids

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"], batch["token_structure_type_ids"], batch["block_pos_ids"] = \
            self.torch_permute_titles(
                batch["input_ids"], batch["token_structure_type_ids"], batch["block_pos_ids"],
                special_tokens_mask=special_tokens_mask
            )
        if self.span_mlm:
            batch["input_ids"] = self.torch_span_mask_tokens(batch["input_ids"],
                                                             special_tokens_mask=special_tokens_mask)

        if self.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

        return batch

    def _cross_merge(self, non_title_segments, title_segments):
        """
        Merge non-title segments and title segments to create a new sequence.

        Example:
            non_title_segments = [non_title_1, non_title_2, non_title_3]
            title_segments = [title_1, title_2]
            merged_result = [non_title_1, title_1, non_title_2, title_2, non_title_3]
        """
        all_segments = []
        len_non_title_segments, len_title_segments = len(non_title_segments), len(title_segments)
        assert len_title_segments == len_non_title_segments or len_non_title_segments - len_title_segments == 1

        for i in range(max(len_title_segments, len_non_title_segments)):
            if non_title_segments:
                all_segments.append(non_title_segments.pop(0))
            if title_segments:
                all_segments.append(title_segments.pop(0))

        return torch.concat(all_segments)

    def torch_permute_titles(self,
                             inputs: Any,
                             token_structure_type_ids: Any,
                             block_pos_ids: Any,
                             special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare permuted inputs/labels for span masked language modeling.

        According to the form linearization discipline, the block title segment directly follows a block type token
        and the title segment ends with another block type token or '<sep>'.
        """
        import torch

        labels = inputs.clone()
        new_block_pos_ids = block_pos_ids.clone()
        permuted_inputs = []
        permuted_token_structure_type_ids = []  # The sequence of token structure type ids will change accordingly.

        # Permute block titles in the linearized form.
        for i in range(labels.size(0)):
            title_segments, non_title_segments = [], []
            structure_type_ids_for_title_segments, structure_type_ids_for_non_title_segments = [], []
            begin_p, j = 0, 0
            find_title = False
            while j < labels[i].size(0) and labels[i][j] != self.pad_token_id:
                if find_title:
                    if labels[i][j] == self.sep_token_id:  # The title part ends.
                        title_segments.append(inputs[i][begin_p:j])
                        structure_type_ids_for_title_segments.append(token_structure_type_ids[i][begin_p:j])
                        find_title = False
                        begin_p = j
                    elif labels[i][j] in self.block_sep_token_ids:
                        # The title part ends and the following title part begins.
                        title_segments.append(inputs[i][begin_p:j])
                        structure_type_ids_for_title_segments.append(token_structure_type_ids[i][begin_p:j])
                        non_title_segments.append(inputs[i][j:j + 1])
                        structure_type_ids_for_non_title_segments.append(token_structure_type_ids[i][j:j + 1])
                        begin_p = j + 1
                else:
                    if labels[i][j] in self.block_sep_token_ids and labels[i][j] != self.sep_token_id:
                        # Find a new title.
                        non_title_segments.append(inputs[i][begin_p:j + 1])
                        structure_type_ids_for_non_title_segments.append(token_structure_type_ids[i][begin_p:j + 1])
                        begin_p = j + 1
                        find_title = True
                j += 1
            # Handle corner case.
            if find_title:
                title_segments.append(inputs[i][begin_p:j])
                structure_type_ids_for_title_segments.append(token_structure_type_ids[i][begin_p:j])
                if j < labels[i].size(0):
                    non_title_segments.append(inputs[i][j:])
                    structure_type_ids_for_non_title_segments.append(token_structure_type_ids[i][j:])
            else:
                non_title_segments.append(inputs[i][begin_p:])
                structure_type_ids_for_non_title_segments.append(token_structure_type_ids[i][begin_p:])

            # Combine title_segments and structure_type_ids_for_non_title_segments together to make sure they are still
            # matching after the shuffling.
            combined = list(zip(title_segments, structure_type_ids_for_title_segments))

            # Permute title segments.
            random.shuffle(combined)

            title_segments = [x[0] for x in combined]
            structure_type_ids_for_title_segments = [x[1] for x in combined]

            # Concatenate segments to form a new sequence.
            if len(non_title_segments) == 0 and len(title_segments) == 0:
                # Extreme case: No block title in the sequence.
                permuted_inputs.append(inputs[i].unsqueeze(0))
                permuted_token_structure_type_ids.append(token_structure_type_ids[i].unsqueeze(0))
            else:
                new_sequence = self._cross_merge(non_title_segments, title_segments)
                new_token_structure_type_ids = self._cross_merge(
                    structure_type_ids_for_non_title_segments, structure_type_ids_for_title_segments)
                assert new_sequence.size(0) == labels[i].size(0)
                assert new_token_structure_type_ids.size(0) == token_structure_type_ids[i].size(0)
                permuted_inputs.append(new_sequence.unsqueeze(0))
                permuted_token_structure_type_ids.append(new_token_structure_type_ids.unsqueeze(0))

        inputs = torch.concat(permuted_inputs, dim=0)
        token_structure_type_ids = torch.concat(permuted_token_structure_type_ids, dim=0)

        # Recover block_pos_ids after title permutation.
        for i in range(block_pos_ids.size(0)):
            block_idx = 0
            for j in range(block_pos_ids[i].size(0)):
                if inputs[i][j] in self.block_sep_token_ids and inputs[i][j] != self.sep_token_id:
                    # Find block seperator.
                    block_idx += 1
                new_block_pos_ids[i][j] = block_idx

        return inputs, labels, token_structure_type_ids, new_block_pos_ids

    def torch_span_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs for span masked language modeling.
        """
        import torch

        # Detects the property separators.
        separators = torch.zeros(inputs.shape).bool()
        for token_id in self.block_sep_token_ids:
            separators = separators | (inputs == token_id)
        separators = separators | (inputs == self.option_sep_token_id)

        # Do masking.
        masked_indices = torch.zeros(inputs.shape)
        for i in range(inputs.size(0)):
            j = 0
            do_mask = False
            while j < inputs[i].size(0) and inputs[i][j] != self.pad_token_id:
                if do_mask:
                    if separators[i][j]:
                        # Reaching a new separator means the end of the current property.
                        # We use Bernoulli(mlm_probability) r.v. to decide whether to mask the next property.
                        if torch.bernoulli(torch.tensor(self.mlm_probability)):
                            do_mask = True
                        else:
                            do_mask = False
                    else:
                        masked_indices[i][j] = 1
                else:
                    if separators[i][j]:
                        # Using Bernoulli(mlm_probability) r.v. to decide whether to mask the current property.
                        if torch.bernoulli(torch.tensor(self.mlm_probability)):
                            do_mask = True
                j += 1

        masked_indices = masked_indices.bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError('Numpy framework is not supported yet!')
