DOCS_PUBLIC_DIR = $(DOCSDIR)/public
DOCS_PUBLIC_SRC_DIR = docs/public
DOCS_PUBLIC_SRCS = $(shell find $(DOCS_PUBLIC_SRC_DIR) -type f -name '*.rst')
DOCS_PUBLIC_SPHINX_BUILDER = html
DOCS_PUBLIC_BUILD_SCRIPT = docs/public/build.sh

docs/public: $(DOCS_PUBLIC_DIR)

.PHONY: foo

$(DOCS_PUBLIC_DIR): $(DOCS_PUBLIC_BUILD_SCRIPT) $(DOCS_PUBLIC_SRCS) python_env foo
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(LIBDIR) \
	PYTHON_ENV=$(PYTHON_ENV) \
	BUILDER=$(DOCS_PUBLIC_SPHINX_BUILDER) \
	SOURCE_DIR=$(DOCS_PUBLIC_SRC_DIR) \
	INSTALL_DIR=$(@) \
	$(DOCS_PUBLIC_BUILD_SCRIPT)

docs/public/publish: docs/public
	rsync --delete -avz  $(DOCS_PUBLIC_DIR)/html/ yyz-webservice-02:/var/www/html/docs/pybuda-docs

docs/pdf: python_env foo
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(LIBDIR) \
	PYTHON_ENV=$(PYTHON_ENV) \
	BUILDER=latexpdf \
	SOURCE_DIR=$(DOCS_PUBLIC_SRC_DIR) \
	INSTALL_DIR=$(@) \
	$(DOCS_PUBLIC_BUILD_SCRIPT)
