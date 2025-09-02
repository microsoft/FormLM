# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Linearize an online form while preserving its structural information."""
from typing import Dict

SPECIAL_TOKEN_LIST = [
    # Block type indicators.
    "<text>",
    "<choice>",
    "<rating>",
    "<likert>",
    "<time>",
    "<date>",
    "<upload>",
    "<description>",
    # New section prompt: We add "<section>" before a new section.
    "<section>",
    # Property separator: We use "<sep>" to separate title, description and body of a component.
    "<sep>",
]

BLOCK_TYPE_NAME_TO_IDX = {
    'textfield': 0,
    'choice': 1,
    'rating': 2,
    'likert': 3,
    'time': 4,
    'date': 5,
    'upload': 6,
    'description': 7,
    'section': 8
}


class FormLinearize:
    """Expects the form having the following format:

    {
        'title': title of the form,
        'description': description of the form,
        'body': [
            # a list of blocks
            {
                'type': type of the block,
                'title': title of the block,
                'description': description of the block
                'options':
                    # For choice type and rating type
                    [a list of option]
                    # For likert type
                    {
                        'rows': [a list of row captions]
                        'columns': [a list of column captions]
                    }
                'selected': whether this sample is included in the experiment dataset for Block Type Suggestion and
                    Next Question Recommendation,
                'choice_selected': whether this sample is included in the experiment dataset for Options Recommendation,
            }
        ]
    }
    """

    def linearize_form(self, form: Dict, include_description=True, add_special_token=True):
        """Converts the form tree into a linearized sequence."""
        linearized_form = [form["title"]]

        description = form["description"]
        if description is not None and len(description) > 0 and include_description:
            linearized_form.append(description)

        linearized_blocks = " ".join(
            [self.linearize_block(block, add_special_token=add_special_token)
             for block in form["body"]])
        linearized_form.append(linearized_blocks)

        if add_special_token:
            linearized_form = " <sep> ".join(linearized_form)
        else:
            linearized_form = " ".join(linearized_form)

        return linearized_form

    def linearize_block(self, block: Dict, include_options=True, add_special_token=True):
        """Converts the block subtree into a linearized sequence."""
        b_type = block["type"]
        b_title = block["title"]
        b_description = block["description"]
        b_options = block["options"] if "options" in block else None

        if add_special_token:
            linearized_block = SPECIAL_TOKEN_LIST[BLOCK_TYPE_NAME_TO_IDX[b_type]] + " " + b_title

            if b_description is not None and len(b_description) > 0:
                linearized_block = linearized_block + " <sep> " + b_description

            if (b_type == "choice" or b_type == "rating") and include_options:
                linearized_block = linearized_block + " <sep> Options: " + "|".join(b_options)
            elif b_type == "likert":
                linearized_block = linearized_block + " <sep> Columns: " + "|".join(b_options['columns']) + \
                                   " Rows: " + "|".join(b_options['rows'])
        else:
            linearized_block = b_title
            if b_description is not None:
                linearized_block = linearized_block + " " + b_description
            if b_options is not None and include_options:
                if b_type == "likert":
                    linearized_block = \
                        linearized_block + " " + " ".join(b_options["columns"]) + " " + " ".join(b_options["rows"])
                else:
                    linearized_block = linearized_block + " " + " ".join(b_options)

        return linearized_block

    def linearize_form_for_option_recommend(self, form: Dict, with_context=True, add_special_token=True):
        """
        Prepare datasets for Options Recommendation.

        Args:
            form: an online form in expected json format.
            with_context: if True, construct the inputs with previous contexts; otherwise, use the block title
                as the input.
            add_special_token: if True, use form linearization technique; otherwise, directly concatenate the previous
                contexts.
        """
        inputs, targets = [], []

        linearized_form = [form["title"]]

        description = form["description"]
        if description is not None and len(description) > 0:
            linearized_form.append(description)

        linearized_blocks = []
        for block in form["body"]:
            if block["type"] != "choice":
                linearized_blocks.append(self.linearize_block(block, add_special_token=add_special_token))
                continue

            linearized_block_without_options = self.linearize_block(block, include_options=False,
                                                                    add_special_token=add_special_token)
            options = block["options"]

            if block["choice_selected"] == "true":
                if with_context:
                    if add_special_token:
                        inputs.append(" <sep> ".join(linearized_form) + " <sep> " + " ".join(
                            linearized_blocks) + " " + linearized_block_without_options + " <sep> Options:")
                    else:
                        inputs.append(" ".join(linearized_form) + " " + " ".join(
                            linearized_blocks) + linearized_block_without_options)
                else:
                    inputs.append(block["title"])
                targets.append("|".join(options))

            linearized_blocks.append(self.linearize_block(block))

        return inputs, targets

    def linearize_form_for_question_recommend(self, form: Dict, with_context=True, add_special_token=True):
        """
        Prepare datasets for Next Question Recommendation.

        Args:
            form: an online form in expected json format.
            with_context: if True, construct the inputs with previous contexts; otherwise, use the block title
                as the input.
            add_special_token: if True, use form linearization technique; otherwise, directly concatenate the previous
                contexts.
        """
        inputs, targets = [], []

        if with_context:
            linearized_form = [form["title"]]

            description = form["description"]
            if description is not None and len(description) > 0:
                linearized_form.append(description)

            linearized_blocks = []
            for block in form["body"]:
                b_type = block["type"]
                if b_type == "section" or b_type == "description" or block["title"] is None or len(block["title"]) == 0:
                    linearized_blocks.append(self.linearize_block(block, add_special_token=add_special_token))
                    continue

                if add_special_token:
                    s = " <sep> ".join(linearized_form) + " <sep> " + " ".join(linearized_blocks) + SPECIAL_TOKEN_LIST[
                        BLOCK_TYPE_NAME_TO_IDX[b_type]]
                else:
                    s = " ".join(linearized_form) + " " + " ".join(linearized_blocks) + " "

                if block["selected"] == "true":
                    inputs.append(s)
                    targets.append(block["title"])

                linearized_blocks.append(self.linearize_block(block, add_special_token=add_special_token))
        else:
            previous_block_title = None
            for block in form["body"]:
                if block["selected"] == "true":
                    if previous_block_title is None:  # For the first question block, the only hint is the form title.
                        inputs.append(form["title"])
                    else:
                        inputs.append(previous_block_title)
                    targets.append(block["title"])
                previous_block_title = block["title"]

        return inputs, targets

    def linearize_form_for_block_type_classification(self, form: Dict, with_context=True, add_special_token=True):
        """
        Prepare datasets for Options Recommendation.

        Args:
            form: an online form in expected json format.
            with_context: if True, construct the inputs with previous contexts; otherwise, use the block title
                as the input. When inputting form context, we use "<mask>" to replace the type attribute of the target
                block to avoid data leakage.
            add_special_token: if True, use form linearization technique; otherwise, directly concatenate the previous
                contexts.
        """
        inputs, labels = [], []
        if with_context:
            linearized_form = [form["title"]]

            description = form["description"]
            if description is not None and len(description) > 0:
                linearized_form.append(description)

            linearized_blocks = []
            for block in form["body"]:
                b_type = block["type"]
                if b_type == 'section':
                    linearized_blocks.append(self.linearize_block(block, add_special_token=add_special_token))
                    continue

                if add_special_token:
                    s = " <sep> ".join(linearized_form) + " <sep> " + " ".join(linearized_blocks) + " <mask> " + \
                        block["title"]
                else:
                    s = " ".join(linearized_form) + " " + " ".join(linearized_blocks) + block["title"]

                if block["selected"] == "true":
                    inputs.append(s)
                    labels.append(BLOCK_TYPE_NAME_TO_IDX[b_type])

                linearized_blocks.append(self.linearize_block(block, add_special_token=add_special_token))
        else:
            for block in form["body"]:
                b_type = block["type"]
                if b_type != "section":
                    if block["selected"] == "true":
                        inputs.append(block["title"])
                        labels.append(BLOCK_TYPE_NAME_TO_IDX[b_type])

        return inputs, labels
