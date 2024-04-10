VENV = (. .venv/bin/activate)

# python3.11
venv:
	python3 -m venv .venv

install: venv
	$(VENV) && pip3 install -r requirements.txt

dataset: install
	$(VENV) && python3 dataset.py oai

install-eval: venv
	curl -L -O https://github.com/openai/evals/archive/refs/heads/main.zip
	unzip -q main.zip
	rm -vf main.zip
	$(VENV) && cd evals-main && pip3 install evals

clean:
	rm -vrf data/
	rm -vrf evals-main *.zip

venv-clean:
	rm -rf .venv/

clean-all: clean venv-clean