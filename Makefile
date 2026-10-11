.PHONY: install test run-yolo run-haar lint clean

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v

run-yolo:
	python3 -m track --source 1

run-haar:
	python3 -m track.run_haar --camera 1

lint:
	python3 -m compileall -q src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
