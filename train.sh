#!/bin/bash

mylist="tlel lr sim"
S="SETUP1 SETUP2 SETUP3 SETUP4 SETUP5"
cols="ns nd nf entropy la ld lt fix ndev age nuc exp rexp sexp"
st="unsampling"

for j in $S; do
    mkdir -p "E:/JIT-DP-experiment/save/$st/$j"
    for i in $mylist; do
        python run.py --model $i \
            --train_data "E:/JIT-VP-Data/FFmpeg/$j/$st/$j-FFmpeg-features-train.jsonl" \
            --test_data "E:/JIT-VP-Data/FFmpeg/$j/$j-FFmpeg-features-test.jsonl" \
            --save_path "save/$st/$j"
        for k in $cols; do
            python E:/NewCrawler/Metrics.py \
                --predict_file "save/$st/$j/$i/${i}_${k}_only_pred_scores.csv" \
                --features_file "E:/JIT-VP-Data/FFmpeg/$j/$j-FFmpeg-features-test.jsonl" \
                --save_folder "save/$st/$j/$i" \
                --model $k
        done
    done
done
