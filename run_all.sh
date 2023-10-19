data_path=$1
save_path=$2
model=$3
parts=("part_1_part_4" "part_3_part_4" "part_4")

for folder in $(find "$data_path" -maxdepth 1 -type d); do
    folder_name=$(basename "$folder")
    echo "folder: $folder_name"
    for part in "${parts[@]}"; do
        python3 run.py --data_path $1 --save_path $2 --train_part "$part" --prj "$folder_name" --model $3
    done
done
