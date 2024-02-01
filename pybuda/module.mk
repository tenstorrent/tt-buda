include pybuda/csrc/module.mk

$(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda.egg-link: python_env $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so
	bash -c "source $(PYTHON_ENV)/bin/activate; cd pybuda; pip install -e ."
	touch -r $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so $@

pybuda: pybuda/csrc $(PYTHON_ENV)/lib/$(PYTHON_VERSION)/site-packages/pybuda.egg-link ;

