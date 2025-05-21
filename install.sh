#!/usr/bin/env bash


system_version="centos";
work_dir="$(pwd)"


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done


if [ $system_version == "centos" ]; then
  python_version=3.9.9

  cd "${work_dir}" || exit 1;
  sh ./script/install_python.sh --system_version "centos" --python_version "${python_version}"

  # source /data/local/bin/TrainerPlatform/bin/activate
  /usr/local/python-${python_version}/bin/pip3 install virtualenv
  mkdir -p /data/local/bin
  cd /data/local/bin || exit 1;
  /usr/local/python-${python_version}/bin/virtualenv TrainerPlatform
fi
