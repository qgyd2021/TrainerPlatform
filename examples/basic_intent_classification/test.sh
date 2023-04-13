#!/usr/bin/env bash


function search_best_ckpt() {
  version="$1";

  last_epoch=$(ls "lightning_logs/${version}/checkpoints" | \
               grep ckpt | awk 'END {print}' | \
               awk -F'[=-]' '/epoch/ {print$2}')
  echo "${last_epoch}"
  target_epoch=$((last_epoch - 5))
  target_file=null
  for file in $(ls "lightning_logs/${version}/checkpoints" | grep ckpt | sort -r):
  do
    this_epoch=$(echo "${file}" | awk -F'[=-]' '/epoch/ {print$2}');

    if [ "${this_epoch}" -le "${target_epoch}" ]; then
      target_file="${file}";
      break;
    fi
  done
  if [ "${target_file}" == null ]; then
    echo "no appropriate file found" && exit 1;
    return 0;
  fi
  echo "${target_file}"
}


search_best_ckpt version_0
