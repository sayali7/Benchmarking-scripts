#!/usr/bin/env bash

PYTHON_CMD="python3 train_cv.py"
DATA_PATH="/media/sayalialatkar/T9/Sayali/FoundationModels/UCE_venv/UCE-main/results/cell_embeddings/extracted_cell_embeddings"
FM_MODEL="scGPT" #scFoundation, scMulan, UCE, Geneformer
MODEL_TYPE="mlp" #logistic

OUT_DIR="/media/sayalialatkar/T9/Sayali/FoundationModels/scBrainLLM/results"
SUMMARY_CSV="${OUT_DIR}/hyperparam_sweep_summary.csv"
mkdir -p "$OUT_DIR"
N_VAL_SPLITS=5

# Hyperparameter grid
BATCH_SIZES=(64 128 256)
LRS=(0.0001 0.001 0.01)
EPOCHS=(25 50 100)
HIDDENS=("256,64" "512,64" "512,128,64")
ACC_STEPS=(0) #(0 2 4)

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LRS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
      for hd in "${HIDDENS[@]}"; do
        for acc in "${ACC_STEPS[@]}"; do
          #echo "Sweep: bs=$bs lr=$lr ep=$ep hd=$hd acc=$acc"
          SUFFIX="bs${bs}_lr${lr}_ep${ep}_hd${hd//,/_}_acc${acc}"
          LOGFILE="${OUT_DIR}/log_${SUFFIX}.txt"
          $PYTHON_CMD \
            --fm_model_name="$FM_MODEL" \
            --model_type="$MODEL_TYPE" \
            --data_path="$DATA_PATH" \
            --batch_size="$bs" \
            --lr="$lr" \
            --epochs="$ep" \
            --hidden_dims="$hd" \
            --accumulation_steps="$acc" \
            --n_splits="$N_VAL_SPLITS" \
            --log_file="$LOGFILE" \
            --out_path="$OUT_DIR" \
            --do_cv

     
          newest=$(ls -t "${OUT_DIR}/cv_summary_"*.csv | head -n1)
          if [ ! -f "$SUMMARY_CSV" ]; then
            cat "$newest" > "$SUMMARY_CSV"
          else
            tail -n +2 "$newest" >> "$SUMMARY_CSV"
          fi

          # Clean up per-run summary
          rm -f "$newest"
        done
      done
    done
  done
done

echo "All sweeps launched."
