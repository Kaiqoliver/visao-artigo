#!/bin/bash

# Pastas Principais
DIR_RAW="/home/koliver/PlantVillage-Dataset/raw/color"
DIR_SEGMENTED="/home/koliver/PlantVillage-Dataset/raw/segmented"
BASE_OUT_DIR="/home/koliver/visao-artigo/results"

# Subpastas para separar as duas abordagens
DIR_FT="$BASE_OUT_DIR/fine_tuning"
DIR_LL="$BASE_OUT_DIR/last_layer"

mkdir -p logs

echo "======================================================"
echo "Iniciando Bateria ÉPICA de Experimentos (14 Runs)"
echo "Início: $(date)"
echo "======================================================"

# ==============================================================================
# FASE 1: FINE-TUNING (Rede inteira treinável)
# ==============================================================================
echo ">>> INICIANDO FASE 1: FINE-TUNING <<<"

# --- Exp 1: Baseline ---
OUT_EXP1="$DIR_FT/exp1_baseline"
mkdir -p $OUT_EXP1
echo "[1/14] FT - Rodando Experimento 1: Baseline..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP1 --use_pretrained True --feature_extract False --aug_geom False --aug_color False --aug_occlus False > logs/ft_exp1_baseline.log 2>&1

# --- Exp 2: Só Geometria ---
OUT_EXP2="$DIR_FT/exp2_geometry"
mkdir -p $OUT_EXP2
echo "[2/14] FT - Rodando Experimento 2: Apenas Geometria..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP2 --use_pretrained True --feature_extract False --aug_geom True --aug_color False --aug_occlus False > logs/ft_exp2_geometry.log 2>&1

# --- Exp 3: Só Cor ---
OUT_EXP3="$DIR_FT/exp3_color"
mkdir -p $OUT_EXP3
echo "[3/14] FT - Rodando Experimento 3: Apenas Cor..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP3 --use_pretrained True --feature_extract False --aug_geom False --aug_color True --aug_occlus False > logs/ft_exp3_color.log 2>&1

# --- Exp 4: Só Oclusão ---
OUT_EXP4="$DIR_FT/exp4_occlusion"
mkdir -p $OUT_EXP4
echo "[4/14] FT - Rodando Experimento 4: Apenas Oclusão..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP4 --use_pretrained True --feature_extract False --aug_geom False --aug_color False --aug_occlus True > logs/ft_exp4_occlusion.log 2>&1

# --- Exp 5: Combo Bruto ---
OUT_EXP5="$DIR_FT/exp5_combo"
mkdir -p $OUT_EXP5
echo "[5/14] FT - Rodando Experimento 5: Combo Bruto..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP5 --use_pretrained True --feature_extract False --aug_geom True --aug_color True --aug_occlus True > logs/ft_exp5_combo.log 2>&1

# --- Exp 6: Segmentado ---
OUT_EXP6="$DIR_FT/exp6_segmented"
mkdir -p $OUT_EXP6
echo "[6/14] FT - Rodando Experimento 6: Segmentado (Baseline)..."
python run.py --dataset_dir $DIR_SEGMENTED --output_dir $OUT_EXP6 --use_pretrained True --feature_extract False --aug_geom False --aug_color False --aug_occlus False > logs/ft_exp6_segmented.log 2>&1

# --- Exp 7: Segmentado + Combo ---
OUT_EXP7="$DIR_FT/exp7_seg_combo"
mkdir -p $OUT_EXP7
echo "[7/14] FT - Rodando Experimento 7: Segmentado + Combo..."
python run.py --dataset_dir $DIR_SEGMENTED --output_dir $OUT_EXP7 --use_pretrained True --feature_extract False --aug_geom True --aug_color True --aug_occlus True > logs/ft_exp7_seg_combo.log 2>&1


# ==============================================================================
# FASE 2: LAST LAYER / FEATURE EXTRACTION (Camadas base congeladas)
# ==============================================================================
echo ">>> INICIANDO FASE 2: LAST LAYER (FEATURE EXTRACTION) <<<"

# --- Exp 8: Baseline ---
OUT_EXP8="$DIR_LL/exp1_baseline"
mkdir -p $OUT_EXP8
echo "[8/14] LL - Rodando Experimento 1: Baseline..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP8 --use_pretrained True --feature_extract True --aug_geom False --aug_color False --aug_occlus False > logs/ll_exp1_baseline.log 2>&1

# --- Exp 9: Só Geometria ---
OUT_EXP9="$DIR_LL/exp2_geometry"
mkdir -p $OUT_EXP9
echo "[9/14] LL - Rodando Experimento 2: Apenas Geometria..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP9 --use_pretrained True --feature_extract True --aug_geom True --aug_color False --aug_occlus False > logs/ll_exp2_geometry.log 2>&1

# --- Exp 10: Só Cor ---
OUT_EXP10="$DIR_LL/exp3_color"
mkdir -p $OUT_EXP10
echo "[10/14] LL - Rodando Experimento 3: Apenas Cor..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP10 --use_pretrained True --feature_extract True --aug_geom False --aug_color True --aug_occlus False > logs/ll_exp3_color.log 2>&1

# --- Exp 11: Só Oclusão ---
OUT_EXP11="$DIR_LL/exp4_occlusion"
mkdir -p $OUT_EXP11
echo "[11/14] LL - Rodando Experimento 4: Apenas Oclusão..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP11 --use_pretrained True --feature_extract True --aug_geom False --aug_color False --aug_occlus True > logs/ll_exp4_occlusion.log 2>&1

# --- Exp 12: Combo Bruto ---
OUT_EXP12="$DIR_LL/exp5_combo"
mkdir -p $OUT_EXP12
echo "[12/14] LL - Rodando Experimento 5: Combo Bruto..."
python run.py --dataset_dir $DIR_RAW --output_dir $OUT_EXP12 --use_pretrained True --feature_extract True --aug_geom True --aug_color True --aug_occlus True > logs/ll_exp5_combo.log 2>&1

# --- Exp 13: Segmentado ---
OUT_EXP13="$DIR_LL/exp6_segmented"
mkdir -p $OUT_EXP13
echo "[13/14] LL - Rodando Experimento 6: Segmentado (Baseline)..."
python run.py --dataset_dir $DIR_SEGMENTED --output_dir $OUT_EXP13 --use_pretrained True --feature_extract True --aug_geom False --aug_color False --aug_occlus False > logs/ll_exp6_segmented.log 2>&1

# --- Exp 14: Segmentado + Combo ---
OUT_EXP14="$DIR_LL/exp7_seg_combo"
mkdir -p $OUT_EXP14
echo "[14/14] LL - Rodando Experimento 7: Segmentado + Combo..."
python run.py --dataset_dir $DIR_SEGMENTED --output_dir $OUT_EXP14 --use_pretrained True --feature_extract True --aug_geom True --aug_color True --aug_occlus True > logs/ll_exp7_seg_combo.log 2>&1


echo "======================================================"
echo "Todos os 14 experimentos foram concluídos!"
echo "Fim: $(date)"
echo "======================================================"