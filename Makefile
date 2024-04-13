VENV = (. .venv/bin/activate)

venv:
	python3 -m venv .venv

create-dataset:
	$(VENV) && pip3 install -r dataset.txt
	$(VENV) && python3 dataset.py

clean:
	rm -vrf *.zip *.csv data/

venv-clean:
	rm -rf .venv/

clean-all: clean venv-clean