#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]] # number of arguments larger then 0
do
key="$1" # after shift the second arg
# echo "$1" # the double quote is parsing the args
# echo '$1' # the single quote is just showling the string

case $key in # dollar sign here?
    -l|--learning-rate)
    LEARNING_RATE="$2"
    shift # shift argument
    shift # shift value
    ;;
    -b|--beta)
    BETA="$2"
    shift # shift argument
    shift # shift value
    ;;
    -p|--keep-prob)
    KEEP_PROB="$2"
    shift # shift argument
    shift # shift value
    ;;
    --default)
    DEFAULT=YES
    shift # shift argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # shift argument
    ;;
esac # end a case statement
done # end a loop
set -- "${POSITIONAL[@]}" # restore positional parameters

echo learning_rate  = "${LEARNING_RATE}"
echo beta           = "${BETA}"
echo keep_probability   = "${KEEP_PROB}"
echo DEFAULT        = "${DEFAULT}"
echo "extra debug: python tensorflow_test.py ${LEARNING_RATE} ${BETA}"

python tensorflow_test.py "${LEARNING_RATE}"

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
    tail -1 "$1"
fi # finish if. also the reverse of if
