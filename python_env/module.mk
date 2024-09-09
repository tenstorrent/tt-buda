# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYTHON_ENV = $(OUT)/python_env
PYTHON_VERSION ?= python3.8

# Each module has a top level target as the entrypoint which must match the subdir name
python_env: $(PYTHON_ENV)/.installed

.PRECIOUS: $(PYTHON_ENV)/.installed $(PYTHON_ENV)/%
$(PYTHON_ENV)/.installed: python_env/requirements.txt
	$(PYTHON_VERSION) -m venv $(PYTHON_ENV)
	bash -c "unset LD_PRELOAD; source build/python_env/bin/activate && pip3 install --upgrade pip"
	bash -c "unset LD_PRELOAD; source $(PYTHON_ENV)/bin/activate && pip3 install wheel==0.37.1"
	bash -c "unset LD_PRELOAD; source $(PYTHON_ENV)/bin/activate && pip3 install -r python_env/requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html"
	touch $@

# Reference for adding PyTorch Geometrics library support. 
# Pausing for now as support for Graph NNs is postponed untill Conv based
# NNs have better support. 
# 
# Note: Think about how installation and build for PyTorch Geometrics (and 
# related dependencies) can be optimized for better performance as it now
# almost doubles the PyBuda build time.
#
# Code reference for section above:
# python_env/requirements_ext.txt
# bash -c "source $(PYTHON_ENV)/bin/activate && pip3 install -r python_env/requirements_ext.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html"

# If you depend on anything (headers, libs, etc) in the python env, build env first
$(PYTHON_ENV)/%: $(PYTHON_ENV)/.installed ;
