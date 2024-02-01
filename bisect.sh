#!/bin/bash

function show_usage(){
    echo "Usage bisect.sh GOOD_REV BAD_REV TEST_COMMAND_0"
    exit 1
}
if [ ! $# -eq 3 ]
then
    show_usage
fi

git fetch

if ! git cat-file -e $1 2> /dev/null
then
  echo "$1 not valid git rev"
  show_usage
fi

if ! git cat-file -e $2 2> /dev/null
then
  echo "$2 not valid git rev"
  show_usage
fi

echo "Running auto_bisect with passing revision: $1, failing revision $2"
echo "Test commands:"

for ((i = 3; i <= $#; i++ )); do
  printf '%s\n' "  ${!i}"
done

read -n1 -s -r -p $'Press c to continue, q to quit\n' key

if [ "$key" = 'q' ]
then
    echo "Exiting"
    exit 0
fi

git bisect start
git bisect bad $2
git bisect good $1
git bisect run bash -c ". compile_and_run_test.sh $3"
git bisect log
git bisect reset