pytest -v pybuda/test/test_user.py \
          pybuda/test/backend/test_silicon.py  \
          pybuda/test/backend/models/test_bert.py::test_ff  \
          pybuda/test/backend/models/test_bert.py::test_pt_encoder \
          --silicon-only

