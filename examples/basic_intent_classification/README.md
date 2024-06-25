## 基础意图分类


```text
source /data/local/bin/TrainerPlatform/bin/activate


sh run.sh \
--system_version "windows" \
--stage 10 \
--stop_stage 10 \
--file_dir "./output_dir_20230414_cn" \
--dataset_filename "../../../datasets/intent_classification/intent_classification_cn.xlsx" \
--pretrained_bert_model_name "chinese-bert-wwm-ext" \
--final_model_name "basic_intent_cn_20230414"


sh run.sh \
--system_version "centos" \
--stage -1 \
--stop_stage 9 \
--file_dir "./output_dir_20230414_en" \
--dataset_filename "../../../datasets/intent_classification/intent_classification_en.xlsx" \
--pretrained_bert_model_name "bert-base-uncased" \
--final_model_name "basic_intent_en_20230414"



sh run.sh \
--system_version "centos" \
--stage -1 \
--stop_stage 9 \
--file_dir "./output_dir_20230413_jp" \
--dataset_filename "../../../datasets/intent_classification/intent_classification_jp.xlsx" \
--pretrained_bert_model_name "bert-base-japanese" \
--final_model_name "basic_intent_jp_20230414"


sh run.sh \
--system_version "centos" \
--stage -1 \
--stop_stage 9 \
--file_dir "./output_dir_20230413_vi" \
--dataset_filename "../../../datasets/intent_classification/intent_classification_vi.xlsx" \
--pretrained_bert_model_name "bert-base-vietnamese-uncased" \
--final_model_name "basic_intent_vi_20230414"




```
