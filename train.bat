@echo off

set "mylist=tlel lr sim"
set "S=SETUP1 SETUP2 SETUP3 SETUP4 SETUP5"
set "cols=ns nd nf entropy la ld lt fix ndev age nuc exp rexp sexp"
set st=unsampling

for %%j in (%S%) do (
	mkdir E:\JIT-DP-experiment\save\%st%\%%j
	for %%i in (%mylist%) do (
    		python run.py --model %%i ^
        	--train_data "E:\JIT-VP-Data\FFmpeg\%%j\%st%\%%j-FFmpeg-features-train.jsonl" ^
        	--test_data "E:\JIT-VP-Data\FFmpeg\%%j\%%j-FFmpeg-features-test.jsonl" ^
		--save_path save\%st%\%%j
		for %%k in (%cols%) do (
			python E:\NewCrawler\Metrics.py ^
			--predict_file "save\%st%\%%j\%%i\%%i_%%k_only_pred_scores.csv" ^
			--features_file "E:\JIT-VP-Data\FFmpeg\%%j\%%j-FFmpeg-features-test.jsonl" ^
			--save_folder save\%st%\%%j\%%i ^
			--model %%k
		)
	)
) 





