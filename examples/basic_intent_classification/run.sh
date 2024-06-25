#!/usr/bin/env bash

# sh run.sh --system_version centos --stage 6 --stop_stage 10 --file_folder_name file_dir_cn --final_model_name basic_intent_cn --dataset_filename /data/tianxing/PycharmProjects/datasets/basic_intent/chinese/intent_classification_cn.xlsx --pretrained_bert_model_name chinese-bert-wwm-ext

# sh run.sh --stage 6 --stop_stage 8
# sh run.sh --stage 8 --stop_stage 8
# sh run.sh --stage 6 --stop_stage 6
# sh run.sh --stage 6 --stop_stage 6

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=9

file_folder_name=file_folder_name
final_model_name=final_model_name
dataset_filename="dataset.xlsx"
nohup_name=nohup.out

pretrained_bert_model_name=bert-base-uncased

train_subset=train.json
valid_subset=valid.json
hierarchical_labels_pkl=hierarchical_labels.pkl
vocabulary=vocabulary
labels_json=labels.json

# cn
#dataset_filename=dataset.xlsx
#pretrained_bert_model_name=chinese-bert-wwm-ext
#final_model_name=basic_intent_cn

# en
#dataset_filename=dataset.xlsx
#pretrained_bert_model_name=bert-base-uncased
#final_model_name=basic_intent_en

# jp
#dataset_filename=dataset.xlsx
#pretrained_bert_model_name=bert-base-japanese
#final_model_name=basic_intent_jp

test_output_filename=test_output.xlsx

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

file_dir="${work_dir}/${file_folder_name}"
final_model_dir="${work_dir}/../../trained_models/${final_model_name}";
pretrained_models_dir="${work_dir}/../../pretrained_models";

mkdir -p "${file_dir}"

$verbose && echo "system_version: ${system_version}"


if [ $system_version == "windows" ]; then
  #source /data/local/bin/TrainerPlatform/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/TrainerPlatform/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/TrainerPlatform/bin/python3'
fi


function search_best_ckpt() {
  version="$1";

  cd "${file_dir}" || exit 1
  last_epoch=$(ls "lightning_logs/${version}/checkpoints" | \
               grep ckpt | \
               awk -F'[=-]' '/epoch/ {print$2}' | \
               sort -n | \
               awk 'END {print}')
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


declare -A pretrained_bert_model_dict
pretrained_bert_model_dict=(
  ["chinese-bert-wwm-ext"]="https://huggingface.co/hfl/chinese-bert-wwm-ext"
  ["bert-base-uncased"]="https://huggingface.co/bert-base-uncased"
  ["bert-base-japanese"]="https://huggingface.co/cl-tohoku/bert-base-japanese"
  ["bert-base-vietnamese-uncased"]="https://huggingface.co/trituenhantaoio/bert-base-vietnamese-uncased"

)
pretrained_model_dir="${pretrained_models_dir}/${pretrained_bert_model_name}"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download pretrained model"

  if [ ! -d "${pretrained_model_dir}" ]; then
    mkdir -p "${pretrained_models_dir}"
    cd "${pretrained_models_dir}" || exit 1;

    repository_url="${pretrained_bert_model_dict[${pretrained_bert_model_name}]}"
    git clone "${repository_url}"

    cd "${pretrained_model_dir}" || exit 1;
    rm flax_model.msgpack && rm pytorch_model.bin && rm tf_model.h5
    wget "${repository_url}/resolve/main/pytorch_model.bin"
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data without irrelevant domain (create train subset, valid subset file)"
  cd "${work_dir}" || exit 1
  python3 1.prepare_data.py \
  --without_irrelevant_domain \
  --dataset_filename "${dataset_filename}" \
  --do_lowercase \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: create hierarchical labels dictionary (create hierarchical_labels.pkl file)"
  cd "${work_dir}" || exit 1
  python3 2.create_hierarchical_labels.py \
  --dataset_filename "${dataset_filename}" \
  --hierarchical_labels_pkl "${file_dir}/${hierarchical_labels_pkl}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: create vocabulary (create vocabulary directory)"
  cd "${work_dir}" || exit 1
  python3 3.create_vocabulary.py \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${file_dir}/${hierarchical_labels_pkl}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --labels_json "${file_dir}/${labels_json}" \

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train model without irrelevant domain"
  cd "${work_dir}" || exit 1
  python3 4.train_model.py \
  --file_dir "${file_dir}" \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${file_dir}/${hierarchical_labels_pkl}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: prepare data with irrelevant domain"
  cd "${work_dir}" || exit 1

  target_file=$(search_best_ckpt version_0);
  test target_file || exit 1;

  python3 1.prepare_data.py \
  --dataset_filename "${dataset_filename}" \
  --do_lowercase \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: train model with irrelevant domain"
  cd "${work_dir}" || exit 1

  target_file=$(search_best_ckpt version_0);
  test target_file || exit 1;

  python3 4.train_model.py \
  --file_dir "${file_dir}" \
  --ckpt_path "${file_dir}/lightning_logs/version_0/checkpoints/${target_file}" \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${file_dir}/${hierarchical_labels_pkl}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: test model"

  target_file=$(search_best_ckpt version_1);
  test target_file || exit 1;

  cd "${work_dir}" || exit 1

  python3 5.test_model.py \
  --ckpt_path "${file_dir}/lightning_logs/version_1/checkpoints/${target_file}" \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${file_dir}/${hierarchical_labels_pkl}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \
  --dataset_filename "${dataset_filename}" \
  --output_filename "${file_dir}/${test_output_filename}"

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  $verbose && echo "stage 7: export model"

  cd "${work_dir}" || exit 1

  target_file=$(search_best_ckpt version_1);
  test target_file || exit 1;

  python3 6.export_model.py \
  --file_dir "${file_dir}" \
  --ckpt_path "${file_dir}/lightning_logs/version_1/checkpoints/${target_file}" \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --hierarchical_labels_pkl "${file_dir}/${hierarchical_labels_pkl}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --train_subset "${file_dir}/${train_subset}" \
  --valid_subset "${file_dir}/${valid_subset}" \

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  $verbose && echo "stage 8: collect files"

  target_file=$(search_best_ckpt version_1);
  test target_file || exit 1;

  cd "${file_dir}" || exit 1

  rm -rf "${final_model_dir}" && mkdir -p "${final_model_dir}";

  cp "pytorch_model.bin" "${final_model_dir}/pytorch_model.bin"
  cp "trace_model.zip" "${final_model_dir}/final.zip"
  cp "labels.json" "${final_model_dir}/labels.json"
  cp -r "vocabulary" "${final_model_dir}/vocabulary"
  cp -f "lightning_logs/version_1/checkpoints/${target_file}" "${final_model_dir}/${target_file}"
  cp "${test_output_filename}" "${final_model_dir}/${test_output_filename}"
  cp "${pretrained_model_dir}/vocab.txt" "${final_model_dir}/vocab.txt"

  # zip
  cd "${final_model_dir}/.." || exit 1;

  if [ -e "${final_model_name}.zip" ]; then
    rm -rf "${final_model_name}_backup.zip"
    mv "${final_model_name}.zip" "${final_model_name}_backup.zip"
  fi
  # zip -r basic_intent_cn.zip basic_intent_cn
#  zip -r "${final_model_name}.zip" "${final_model_name}"
#  rm -rf "${final_model_name}"
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  $verbose && echo "stage 9: test jit model"
  cd "${work_dir}" || exit 1

  python3 7.test_jit_model.py \
  --model_dir "${final_model_dir}" \
  --pretrained_model_dir "${pretrained_model_dir}"
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  $verbose && echo "stage 10: clear file_dir"
  cd "${work_dir}" || exit 1

  rm -rf "${file_dir}";
  rm -rf "${nohup_name}";
fi
