# KNN - word / text correction

## Checkpoint
### Dataset
As our dataset for evaluation we have used [Korpus SYN2015](https://wiki.korpus.cz/doku.php/cnk:syn2015#korpus_syn2015)

#### Generate spolied words
```console
$ make checkpoint-dataset
Output file: '$PWD/data/word-spoiler/samples.jsonl'
```

#### Install evaluation framework
```console
$ make install-eval
```

#### Evaluate dataset
```console
$ source .venv/bin/activate
(.venv) $ export OPENAI_API_KEY=<your-key>
(.venv) $ oaieval gpt-3.5-turbo word-spoiler --max_samples 10000 --registry_path .
```

## Experiments