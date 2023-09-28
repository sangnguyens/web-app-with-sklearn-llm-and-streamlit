install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
	
test:
	python -m pytest -vv test.py

format:
	black src/*.py  *.py

lint:
	pylint --disable=R,C *.py

container-lint:
	# docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	streamlit run --server.port 4000 src/app.py

all: install lint test format deploy
