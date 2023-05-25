#!/bin/bash

COMMAND="$(python process_args.py "$@")"

echo "${COMMAND}"

NL='
'
case $COMMAND in
  *"$NL"*) ;;
        *) bash -c "${COMMAND}" ;;
esac
