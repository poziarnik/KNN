# KNN - word / text correction

## Dataset
As our dataset for evaluation we have used [Korpus SYN2015](https://wiki.korpus.cz/doku.php/cnk:syn2015#korpus_syn2015)

### Install evaluation framework
```console
$ make install-eval
```

### Generate spolied words
```console
$ make dataset
Output file: 'data/word-spoiler.jsonl'
```

### Evaluate dataset
```console
$ source .venv/bin/activate
(.venv) $ export OPENAI_API_KEY=<your-key>
(.venv) $ oaieval gpt-3.5-turbo word-spoiler --max_samples 10000 --registry_path .
```