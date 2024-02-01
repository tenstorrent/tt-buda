
# How to run pybuda on silicon

## Docker

To create a docker, run `bin/run-docker.sh` from `third_party/budabackend`. Then, enter it use `docker exec -u $USER -it special-$USER bash`.

## Env

Grayskull and Wormhole machines require slightly different setups, so there are setup scripts for each. To build everything run the appropriate script:

* `source env_for_silicon.sh` (Grayskull) 

or 

* `source env_for_wormhole.sh` (Wormhole)

Now, you should be able to run pybuda pytests and python scripts.

## Run

For example, try:

`pytest -svv pybuda/test/backend/test_silicon.py::test_basic_training[Grayskull-acc1-mbc1-microbatch1-s1]`

