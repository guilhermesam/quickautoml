#!/bin/bash
RED="\e[031m"
GREEN="\e[32m"
RESET="\e[39m"
PROJECT_ROOT="./"

has_param() {
  local term="$1"
  shift
  for arg; do
      if [[ $arg == "$term" ]]; then
          return 0
      fi
  done
  return 1
}

if has_param '-d' "$@"; then
  ENV="development"
else
  ENV="production"
fi

while getopts ":p:" opt; do
  case $opt in
    p)
      PACKAGE_NAME=$OPTARG
      ;;
    *)
  esac
done

# pegar ultima versao no pip
echo -e "Installing ${GREEN}$PACKAGE_NAME${RESET} with ${GREEN}$ENV${RESET} environment"

# Transform this verification in a function
if [ $ENV == "development" ]; then
  touch dev-requirements.txt
  if grep -Fxq "$PACKAGE_NAME" dev-requirements.txt
  then
    echo -e "${RED}Error:${RESET} Package ${RED}${PACKAGE_NAME}${RESET} is already installed!"
    exit 1
  fi
  echo "$PACKAGE_NAME" >> "$PROJECT_ROOT/dev-requirements.txt"
else
  touch requirements.txt
  if grep -Fxq "$PACKAGE_NAME" requirements.txt
  then
    echo -e "${RED}Error:${RESET} Package ${RED}${PACKAGE_NAME}${RESET} is already installed!"
    exit 1
  fi
  echo "$PACKAGE_NAME" >> "$PROJECT_ROOT/requirements.txt"
fi

pip install --quiet "$PACKAGE_NAME"

echo -e "${GREEN}DONE!$RESET"
