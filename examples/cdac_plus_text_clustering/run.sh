#!/usr/bin/env bash

# sh run.sh --stage -2 --stop_stage 5 --system_version windows
# sh run.sh --stage 4 --stop_stage 5 --system_version centos
# sh run.sh --stage 0 --stop_stage 0 --system_version centos
# sh run.sh --stage 1 --stop_stage 1 --system_version centos
# sh run.sh --stage 2 --stop_stage 2 --system_version centos
# sh run.sh --stage 2 --stop_stage 5 --system_version centos
# sh run.sh --stage 6 --stop_stage 6 --system_version centos

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=9

work_dir="$(pwd)"
file_folder_name=file_folder_name

basic_intent_classification_datasets_dir=/data/tianxing/PycharmProjects/TrainerPlatform/datasets
intent_classification_xlsx=/data/tianxing/PycharmProjects/TrainerPlatform/datasets/waba_intent_classification.xlsx

train_labeled_subset=train_labeled.json
valid_labeled_subset=valid_labeled.json
train_all_subset=train_all.json
dataset_excel=dataset.xlsx
vocabulary=vocabulary
all_vector=all_vector.json

pretrained_bert_model_name=bert-base-uncased

n_clusters=200
k_classes=14

with_classification_cross_training=true


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
mkdir -p "${file_dir}"

pretrained_models_dir="${work_dir}/../../pretrained_models";

classification_serialization_dir="${file_dir}/classification"
pretrain_serialization_dir="${file_dir}/pretrain"
finetune_serialization_dir="${file_dir}/finetune"


$verbose && echo "system_version: ${system_version}"

if [ $system_version == "windows" ]; then
  #source /data/local/bin/TrainerPlatform/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/TrainerPlatform/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/TrainerPlatform/bin/python3'
fi


declare -A pretrained_bert_model_dict
pretrained_bert_model_dict=(
  ["chinese-bert-wwm-ext"]="https://huggingface.co/hfl/chinese-bert-wwm-ext"
  ["bert-base-uncased"]="https://huggingface.co/bert-base-uncased"
  ["bert-base-japanese"]="https://huggingface.co/cl-tohoku/bert-base-japanese"
  ["bert-base-vietnamese-uncased"]="https://huggingface.co/trituenhantaoio/bert-base-vietnamese-uncased"

)
pretrained_model_dir="${pretrained_models_dir}/${pretrained_bert_model_name}"


if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  $verbose && echo "stage -2: download pretrained model"

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


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download datasets"
  cd "${work_dir}" || exit 1;

  mkdir -p "${basic_intent_classification_datasets_dir}" && cd "${basic_intent_classification_datasets_dir}" || exit 1;

  dataset_name_array=(
    intent_classification_cn
    intent_classification_en
    intent_classification_jp
    intent_classification_vi
  )

  for dataset_name in ${dataset_name_array[*]}
  do
    if [ ! -d "${dataset_name}" ]; then
      wget -c "https://huggingface.co/datasets/qgyd2021/basic_intent_classification/resolve/main/${dataset_name}.xlsx"
    fi
  done
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: prepare data"
  cd "${work_dir}" || exit 1
  python3 1.prepare_data.py \
  --intent_classification_xlsx "${intent_classification_xlsx}" \
  --train_labeled "${file_dir}/${train_labeled_subset}" \
  --valid_labeled "${file_dir}/${valid_labeled_subset}" \
  --train_all "${file_dir}/${train_all_subset}" \
  --dataset_excel "${file_dir}/${dataset_excel}" \

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: make vocabulary"
  cd "${work_dir}" || exit 1
  python3 2.make_vocabulary.py \
  --intent_classification_xlsx "${intent_classification_xlsx}" \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --train_all "${file_dir}/${train_all_subset}" \
  --vocabulary "${file_dir}/${vocabulary}" \

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train classification"
  cd "${work_dir}" || exit 1

  if [ ${with_classification_cross_training} != true ]; then
    python3 3.train_classification.py \
    --pretrained_model_dir "${pretrained_model_dir}" \
    --train_labeled "${file_dir}/${train_labeled_subset}" \
    --valid_labeled "${file_dir}/${valid_labeled_subset}" \
    --train_all "${file_dir}/${train_all_subset}" \
    --vocabulary "${file_dir}/${vocabulary}" \
    --n_clusters ${n_clusters} \
    --serialization_dir "${classification_serialization_dir}" \

  fi

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: train pretrain"
  cd "${work_dir}" || exit 1

  if [ ${with_classification_cross_training} != true ]; then

    python3 4.train_pretrain.py \
    --pretrained_model_dir "${pretrained_model_dir}" \
    --train_labeled "${file_dir}/${train_labeled_subset}" \
    --valid_labeled "${file_dir}/${valid_labeled_subset}" \
    --train_all "${file_dir}/${train_all_subset}" \
    --vocabulary "${file_dir}/${vocabulary}" \
    --n_clusters ${n_clusters} \
    --k_classes ${k_classes} \
    --serialization_dir "${pretrain_serialization_dir}" \
    --pretrain_model_filename "${classification_serialization_dir}/best.bin" \

  else

    python3 4.train_pretrain.py \
    --pretrained_model_dir "${pretrained_model_dir}" \
    --train_labeled "${file_dir}/${train_labeled_subset}" \
    --valid_labeled "${file_dir}/${valid_labeled_subset}" \
    --train_all "${file_dir}/${train_all_subset}" \
    --vocabulary "${file_dir}/${vocabulary}" \
    --n_clusters ${n_clusters} \
    --k_classes ${k_classes} \
    --serialization_dir "${pretrain_serialization_dir}" \
    --with_classification_cross_training ${with_classification_cross_training} \

  fi

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 4: train finetune"
  cd "${work_dir}" || exit 1
  python3 5.train_finetune.py \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --train_labeled "${file_dir}/${train_labeled_subset}" \
  --valid_labeled "${file_dir}/${valid_labeled_subset}" \
  --train_all "${file_dir}/${train_all_subset}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --n_clusters ${n_clusters} \
  --k_classes ${k_classes} \
  --serialization_dir "${finetune_serialization_dir}" \
  --pretrain_model_filename "${pretrain_serialization_dir}/best.bin" \

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: gen vector"
  cd "${work_dir}" || exit 1
  python3 6.gen_vector.py \
  --train_all "${file_dir}/${train_all_subset}" \
  --vocabulary "${file_dir}/${vocabulary}" \
  --all_vector "${file_dir}/${all_vector}" \
  --n_clusters ${n_clusters} \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --pretrain_model_filename "${finetune_serialization_dir}/best.bin" \

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  $verbose && echo "stage 6: faiss test"
  cd "${work_dir}" || exit 1
  python3 7.faiss_test.py \
  --vocabulary "${file_dir}/${vocabulary}" \
  --all_vector "${file_dir}/${all_vector}" \
  --n_clusters ${n_clusters} \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --pretrain_model_filename "${finetune_serialization_dir}/best.bin" \

fi
