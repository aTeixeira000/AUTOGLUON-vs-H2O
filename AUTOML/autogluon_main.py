from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score  # métricas

# ====================== CONFIG ======================

BASE_DIR = Path(__file__).resolve().parent
LABEL = 'Outcome'
TRAIN_PATH = 'diabetes.csv'
SAVE_PATH  = BASE_DIR / 'agModels-predictClass'
OUT_PRED_CSV = BASE_DIR / 'leaderboards' / 'autogluon_leaderboard.csv'

# ----------------- leitura -----------------
def ler_treino() -> pd.DataFrame:
    # TabularDataset se comporta como um DataFrame para o que precisamos
    return TabularDataset(str(TRAIN_PATH))

# ----------------- treino -----------------
def train_predictor(train_df: pd.DataFrame) -> TabularPredictor:
    predictor = TabularPredictor(
        eval_metric='f1',
        label=LABEL,
        path=SAVE_PATH
    ).fit(train_df, num_bag_folds=10)
    return predictor

# ----------------- leaderboard + métricas -----------------
def show_leaderboard(
        predictor: TabularPredictor,
        data_df: pd.DataFrame | None = None,
        out_csv: OUT_PRED_CSV = None,
) -> pd.DataFrame:
    # leaderboard original do AutoGluon
    lb = predictor.leaderboard(data=data_df, silent=True)

    # ground truth
    y_true = data_df[LABEL].values

    acc_list = []
    rec_list = []
    f1_list  = []

    # calcular métricas modelo a modelo
    for model_name in lb['model']:
        y_pred = predictor.predict(data_df, model=model_name)
        acc_list.append(accuracy_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        f1_list.append(f1_score(y_true, y_pred))

    # montar dataframe final só com tempos + métricas
    out_df = pd.DataFrame()
    out_df['model'] = lb['model']

    # tempo de treino (fit_time é o mais comum)
    if 'fit_time' in lb.columns:
        out_df['train_time'] = lb['fit_time']
    elif 'fit_time_marginal' in lb.columns:
        out_df['train_time'] = lb['fit_time_marginal']
    else:
        out_df['train_time'] = None  # fallback

    # tempo de predição (validação ou teste)
    if 'pred_time_val' in lb.columns:
        out_df['pred_time'] = lb['pred_time_val']
    elif 'pred_time_test' in lb.columns:
        out_df['pred_time'] = lb['pred_time_test']
    else:
        out_df['pred_time'] = None  # fallback

    # métricas
    out_df['accuracy'] = acc_list
    out_df['recall']   = rec_list
    out_df['f1']       = f1_list

    out_df.to_csv(str(out_csv), index=False)
    return out_df

# ----------------- main -----------------
def main():
    train_df = ler_treino()
    predictor = train_predictor(train_df)
    show_leaderboard(predictor, data_df=train_df, out_csv=OUT_PRED_CSV)

if __name__ == "__main__":
    main()
