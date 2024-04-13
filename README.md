# KNN - word / text correction

## Base-line
### Dataset
As our dataset we are using texts from czech wikipedia.

#### Generate text with errors
 - `50%` of the dataset contains text with introduced errors
 - `30%` of the text has some errors
```console
$ make create-dataset
Introducing Errors: 100%|██████████████████████████| 25203/25203 [00:02<00:00, 12192.85it/s]
Saving dataset...
```

#### Evaluate GPT-3 on generated data
 - send API calls to GPT-3 to correct the provided text
```console
$ make evaluate-dataset
Loading dataset...
Randomizing dataset...
Sending API calls to GPT.: 100%|██████████████████████████| 50/50 [00:41<00:00,  1.21it/s]
Saving evaluation data...
```

## Experiments