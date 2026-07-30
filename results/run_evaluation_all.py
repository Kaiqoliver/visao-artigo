import os
import pandas as pd
from evaluate_cross import evaluate_model

BASE_RESULTS_DIR = "/home/koliver/visao-artigo/results"

def main():
    # Mapeamento limpo para as tags do artigo
    eval_targets = [
        # (Caminho Relativo, Modo de Treino, Nome Amigável para a Tabela)
        ("fine_tuning/exp1_baseline", "Fine-Tuning", "1. Baseline"),
        ("fine_tuning/exp2_geometry", "Fine-Tuning", "2. Geometria"),
        ("fine_tuning/exp3_color", "Fine-Tuning", "3. Cor/Blur"),
        ("fine_tuning/exp4_occlusion", "Fine-Tuning", "4. Oclusão"),
        ("fine_tuning/exp5_combo", "Fine-Tuning", "5. Combo Bruto"),
        ("fine_tuning/exp6_segmented", "Fine-Tuning", "6. Segmentado"),
        ("fine_tuning/exp7_seg_combo", "Fine-Tuning", "7. Seg. + Combo"),
        
        ("last_layer/exp1_baseline", "Last Layer", "1. Baseline"),
        ("last_layer/exp2_geometry", "Last Layer", "2. Geometria"),
        ("last_layer/exp3_color", "Last Layer", "3. Cor/Blur"),
        ("last_layer/exp4_occlusion", "Last Layer", "4. Oclusão"),
        ("last_layer/exp5_combo", "Last Layer", "5. Combo Bruto"),
        ("last_layer/exp6_segmented", "Last Layer", "6. Segmentado"),
        ("last_layer/exp7_seg_combo", "Last Layer", "7. Seg. + Combo"),
    ]

    print("======================================================")
    print("Iniciando Avaliação Cross-Dataset em Lote (PlantDoc)")
    print("======================================================")

    summary_records = []

    for rel_path, training_mode, exp_name in eval_targets:
        exp_dir = os.path.join(BASE_RESULTS_DIR, rel_path)
        model_path = os.path.join(exp_dir, "model.pth")
        output_csv = os.path.join(exp_dir, "plantdoc_cross_eval.csv")

        if os.path.exists(model_path):
            print(f"\n--- Processando: [{training_mode}] {exp_name} ---")
            
            # Define se usou feature extraction (True para Last Layer, False para Fine-Tuning)
            f_ext = (training_mode == "Last Layer")
            
            success = evaluate_model(model_path, output_csv, feature_extract=f_ext)
            
            if success and os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                corrects = (df['True_PlantDoc'] == df['Predicted_PlantDoc']).sum()
                total = len(df)
                acc = (corrects / total) * 100 if total > 0 else 0
                
                summary_records.append({
                    "Training_Strategy": training_mode,
                    "Experiment": exp_name,
                    "Total_Evaluated": total,
                    "Correct_Predictions": corrects,
                    "PlantDoc_Accuracy_Pct": round(acc, 2)
                })
        else:
            print(f"\n[IGNORADO] Modelo não encontrado em: {model_path}")

    if not summary_records:
        print("\n[AVISO] Nenhum modelo foi avaliado com sucesso. Verifique se os treinamentos terminaram.")
        return

    # Constrói o DataFrame mestre consolidado
    df_summary = pd.DataFrame(summary_records)
    
    # Salva na raiz da pasta results
    master_csv_path = os.path.join(BASE_RESULTS_DIR, "summary_ablation_results.csv")
    df_summary.to_csv(master_csv_path, index=False)
    
    # Exibe a tabela bonita no terminal
    print("\n" + "="*70)
    print("📊 TABELA CONSOLIDADA DE ABLAÇÃO (Cross-Dataset PlantDoc)")
    print("="*70)
    print(df_summary.to_string(index=False))
    print("="*70)
    print(f"Arquivo mestre salvo com sucesso em: {master_csv_path}")

    # BÔNUS: Gera uma tabela pivô ou formato LaTeX amigável para o artigo
    print("\n📝 Código LaTeX gerado automaticamente para o seu artigo:")
    print("-" * 50)
    
    # Reorganiza para formato comparativo (linhas = Experimentos, Colunas = Estratégias)
    pivot_df = df_summary.pivot(index="Experiment", columns="Training_Strategy", values="PlantDoc_Accuracy_Pct")
    print(pivot_df.to_latex(float_format="%.2f"))
    print("-" * 50)

if __name__ == "__main__":
    main()