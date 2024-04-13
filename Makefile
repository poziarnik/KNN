VENV = (. .venv/bin/activate)

venv:
	python3 -m venv .venv

install-deps: venv
	$(VENV) && pip3 install -r requirements.txt

create-dataset:
	$(VENV) && python3 dataset.py

evaluate-dataset:
	$(VENV) && python3 evaluate.py

clean:
	rm -vrf *.zip *.csv data/

venv-clean:
	rm -rf .venv/

clean-all: clean venv-clean