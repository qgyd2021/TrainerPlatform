#!/usr/bin/env bash

# sh run.sh --stage -1 --stop_stage 9
# sh run.sh --stage -1 --stop_stage 5 --system_version centos --file_folder_name task_cnn_voicemail_id_id --final_model_name cnn_voicemail_id_id
# sh run.sh --stage 3 --stop_stage 4
# sh run.sh --stage 4 --stop_stage 4
# sh run.sh --stage 3 --stop_stage 3 --system_version centos --file_folder_name task_cnn_voicemail_id_id

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=9

work_dir="$(pwd)"
file_folder_name=file_folder_name
final_model_name=final_model_name
filename_patterns="/data/tianxing/PycharmProjects/datasets/voicemail/id-ID/wav_finished/*/*.wav"
nohup_name=nohup.out

# model params
batch_size=64
max_epochs=200
save_top_k=10
patience=5


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

file_dir="${work_dir}/${file_folder_name}"
final_model_dir="${work_dir}/../../trained_models/${final_model_name}";


$verbose && echo "system_version: ${system_version}"
$verbose && echo "file_folder_name: ${file_folder_name}"

if [ $system_version == "windows" ]; then
  #source /data/local/bin/TrainerPlatform/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/TrainerPlatform/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/TrainerPlatform/bin/python3'
fi


function search_best_ckpt() {
  version="$1";
  patience="$2";

  cd "${file_dir}" || exit 1
  last_epoch=$(ls "lightning_logs/${version}/checkpoints" | \
               grep ckpt | awk 'END {print}' | \
               awk -F'[=-]' '/epoch/ {print$2}')
  target_epoch=$((last_epoch - patience))
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


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1
  python3 1.prepare_data.py \
  --file_dir "${file_dir}" \
  --filename_patterns "${filename_patterns}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: create vocabulary"
  cd "${work_dir}" || exit 1
  python3 2.create_vocabulary.py \
  --file_dir "${file_dir}"

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"
  cd "${work_dir}" || exit 1
  python3 3.train_model.py \
  --file_dir "${file_dir}" \
  --batch_size ${batch_size} \
  --max_epochs ${max_epochs} \
  --save_top_k ${save_top_k} \
  --patience "${patience}" \
  --train_dataset "${file_dir}/train.xlsx" \
  --test_dataset "${file_dir}/test.xlsx"

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: test model"

  target_file=$(search_best_ckpt version_0 "${patience}");
  test target_file || exit 1;

  cd "${work_dir}" || exit 1

  python3 4.test_model.py \
  --file_dir "${file_dir}" \
  --ckpt_path "lightning_logs/version_0/checkpoints/${target_file}" \
  --train_dataset "${file_dir}/train.xlsx" \
  --test_dataset "${file_dir}/test.xlsx"

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: export model"

  target_file=$(search_best_ckpt version_0 "${patience}");
  test target_file || exit 1;

  cd "${work_dir}" || exit 1

  python3 5.export_model.py \
  --file_dir "${file_dir}" \
  --ckpt_path "lightning_logs/version_0/checkpoints/${target_file}" \

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: collect files"

  mkdir -p "${final_model_dir}";

  target_file=$(search_best_ckpt version_0 "${patience}");
  test target_file || exit 1;

  cd "${work_dir}" || exit 1;

  cp "${file_dir}/evaluation.xlsx" "${final_model_dir}/evaluation.xlsx"
  cp "${file_dir}/pytorch_model.bin" "${final_model_dir}/pytorch_model.bin"
  cp "${file_dir}/cnn_voicemail.pth" "${final_model_dir}/cnn_voicemail.pth"
  cp "${file_dir}/trace_model.zip" "${final_model_dir}/trace_model.zip"
  cp "${file_dir}/trace_quant_model.zip" "${final_model_dir}/trace_quant_model.zip"
  cp "${file_dir}/script_model.zip" "${final_model_dir}/script_model.zip"
  cp "${file_dir}/script_quant_model.zip" "${final_model_dir}/script_quant_model.zip"
  cp -r "${file_dir}/vocabulary" "${final_model_dir}/vocabulary"

  cd "${final_model_dir}/.." || exit 1;

  if [ -e "${final_model_name}.zip" ]; then
    rm -rf "${final_model_name}_backup.zip"
    mv "${final_model_name}.zip" "${final_model_name}_backup.zip"
  fi
  # zip -r cnn_voicemail_zh_tw.zip cnn_voicemail_zh_tw
  zip -r "${final_model_name}.zip" "${final_model_name}"
  rm -rf "${final_model_name}"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: clear file_dir"
  cd "${work_dir}" || exit 1

  rm -rf "${file_dir}";
  rm -rf "${nohup_name}";
fi
