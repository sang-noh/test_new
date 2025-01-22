.PHONY: run test

run:
	python json_reader.py

test:
	pytest .

install:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__ *.pyc .pytest_cache
