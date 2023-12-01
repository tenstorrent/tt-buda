# Makefile
SHELL := /usr/bin/bash

# Help
.PHONY: help
help:
	@echo "Commands:"
	@echo "style    : executes style formatting."
	@echo "clean    : cleans all unnecessary files."
	@echo "clean_tt : cleans all unnecessary Tenstorrent model files."


# Styling
.ONESHELL:
.PHONY: style
style:
	@echo "Running styling..."
	@black .
	@flake8
	@python3 -m isort .

# Cleaning
.ONESHELL:
.PHONY: clean
clean: style
	@echo "Cleaning files..."
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	@find . | grep -E ".pytest_cache" | xargs rm -rf
	@find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	@find . | grep -E ".trash" | xargs rm -rf
	@rm -f .coverage
	@echo "All done cleaning!"

# Cleaning PyBUDA artifacts
.ONESHELL:
.PHONY: clean_tt
clean_tt:
	@echo "Cleaning TT files..."
	@find . | grep -E ".hlkc_cache" | xargs rm -rf
	@find . | grep -E "*netlist.yaml" | xargs rm -rf
	@find . | grep -E ".pkl_memoize_py3" | xargs rm -rf
	@find . | grep -E "generated_modules" | xargs rm -rf
	@find . | grep -E "tt_build" | xargs rm -rf
	@echo "All done cleaning TT files!"
