VENV = (. .venv/bin/activate)

venv:
	python3 -m venv .venv

install: venv
	pip install -r requirements.txt

eval:
	curl -L -O https://github.com/openai/evals/archive/refs/heads/main.zip
	unzip -q main.zip
	rm -vf main.zip

clean:
	rm -vrf evals-main *.zip