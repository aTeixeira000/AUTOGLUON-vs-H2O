from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from catboost.utils import eval_metric
from sklearn.metrics import accuracy_score, recall_score, f1_score  # <--- NOVO IMPORT

# ====================== CONFIG ======================

BASE_DIR = Path(__file__).resolve().parent
LABEL = 'Outcome'
TRAIN_PATH = 'diabetes.csv'
SAVE_PATH  = BASE_DIR / 'agModels-predictClass'
OUT_PRED_CSV = BASE_DIR / 'leaderboards' / 'autogluon_leaderboard.csv'

# ----------------- leitura -----------------
def ler_treino() -> pd.DataFrame:
    return TabularDataset(str(TRAIN_PATH))

# ----------------- treino -----------------
def train_predictor(train_df: pd.DataFrame) -> TabularPredictor:
    predictor = TabularPredictor(eval_metric = 'f1',label=LABEL, path=SAVE_PATH).fit(train_df, num_bag_folds=10)
    return predictor

# ----------------- leaderboard -----------------
def show_leaderboard(
        predictor: TabularPredictor,
        data_df: pd.DataFrame | None = None,
        out_csv: OUT_PRED_CSV = None,
) -> pd.DataFrame:
    # leaderboard original do AutoGluon
    lb = predictor.leaderboard(data=data_df, silent=True)

    # define as colunas que você quer manter
    cols_to_keep = [
        'model',
        'score_test',
        'score_val',
        'eval_metric',
        'pred_time_test',
        'pred_time_val',
    ]
    # algumas colunas podem não existir dependendo do treino,
    # então filtramos só as que existem de fato
    cols_existentes = [c for c in cols_to_keep if c in lb.columns]

    lb_reduzido = lb[cols_existentes]

    lb_reduzido.to_csv(str(out_csv), index=False)
    return lb_reduzido

# ----------------- main -----------------
def main():
    train_df = ler_treino()
    predictor = train_predictor(train_df)
    show_leaderboard(predictor, data_df=train_df, out_csv=OUT_PRED_CSV)

if __name__ == "__main__":
    main()
