Tenstorrent PyBuda
==================

[INTERNAL README]

`pybuda` is a python front-end for Tenstorrent Buda framework, and the main environment for writing and running ML models.

`pybuda` is designed to be used externally, and therefore many configuration defaults are for "production" mode - debug logging is turned off, device defaults to silicon, etc.
When used for internal develoment, set env variable PYBUDA_DEVMODE to a non-empty string, which will switch defaults to more development-friendly options. 

Build
-----
After cloning and recursively updating the repo:
```
git submodule sync
git submodule update --init --recursive -f
```

Run `make pybuda`, and enable the python environment by running `source build/python_env/bin/activate`
Optionally, run `CONFIG=debug make build` to build with debug information.

To build and use tvm do:
```
./third_party/tvm/install.sh
source ./third_party/tvm/enable.sh
```

In our dev environment tests need to run from the base dir `pytest pybuda/test/test_sanity.py` instead of `cd pybuda/test && pytest test_sanity.py`, the second option will not work.

Git Workflow
------------
In general we prefer to use rebasing instead of merging of development/feature branches.  The suggested git workflow is as follows:
```
git checkout -b my_feature_branch
# Do some work on my_feature_branch, now ready to merge into main
git checkout main
git pull
git checkout my_feature_branch
git rebase main
# resolve merge conflicts
git push --force # optionally update your remote my_feature_branch
git checkout main
git merge my_feature_branch
```
