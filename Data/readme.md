# Open Online Form Dataset

Open Online Form (OOF) Dataset contains 62K public online forms in json format. Specifically, there are 49904 forms in the training set, 6238 forms in the dev set, and 6238 forms in the test set.

The json file follows the following format:

```
{
    'title': title of the form,
    'description': description of the form,
    'body': [
        # a list of blocks
        {
            'type': type of the block, # textfield / choice / rating / likert / time / date / upload / description
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
        }
    ]
}
```

For tasks of Form Creation Aids in our paper, we do random sampling in each form to construct our downstream tasks datasets. We include two more keys in each block:
- `selected: 'true'` means this block is selected in Next Question Recommendation and Block Type Suggestion
- `choice_selected: 'true'` means this *Choice* block is selected in Options Recommendation.
