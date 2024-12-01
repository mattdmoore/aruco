.PHONY: run

setup:
	poetry install

run:
	poetry run python3 -m aruco.main
