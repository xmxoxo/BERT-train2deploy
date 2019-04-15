#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动BERT分类模型 

echo 'start BERT mobile svr...'
cd /mnt/sda1/transdat/bert-demo/bert/mobile_svr
sudo rm -rf tmp*

export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -mode CLASS \
    -max_seq_len 128 \
    -http_port 8091 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1 
