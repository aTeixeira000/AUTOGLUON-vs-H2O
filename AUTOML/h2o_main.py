from pathlib import Path

import pandas as pd
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
from sklearn.metrics import accuracy_score, recall_score, f1_score  # <--- MÉTRICAS

# ====================== CONFIG ======================
BASE_DIR = Path(__file__).resolve().parent
LABEL = 'Outcome'
TRAIN_PATH = 'diabetes.csv'
LEADERBOARD_CSV = BASE_DIR / 'leaderboards' / 'h2o_leaderboard.csv'


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def treinar_modelo(train_pdf: pd.DataFrame):

    train_h2o = h2o.H2OFrame(train_pdf)
    train_h2o[LABEL] = train_h2o[LABEL].asfactor()
    y = LABEL
    x = [c for c in train_h2o.col_names if c != y]

    aml = H2OAutoML(
        nfolds=10,
        max_models=30,
        seed=42
    )
    aml.train(x=x, y=y, training_frame=train_h2o)

    return aml


def salvar_leaderboard(aml, train_pdf: pd.DataFrame, out_csv: str = LEADERBOARD_CSV):
    # Leaderboard do H2O com colunas extras de tempo
    lb_h2o = get_leaderboard(aml, extra_columns="ALL")
    leaderboard_df = lb_h2o.as_data_frame()  # já vem com training_time_ms, predict_time_per_row_ms

    # Conjunto em que você quer avaliar (aqui: treino)
    train_h2o = h2o.H2OFrame(train_pdf)
    train_h2o[LABEL] = train_h2o[LABEL].asfactor()

    y_true = train_pdf[LABEL].values

    acc_list = []
    rec_list = []
    f1_list = []

    # Para cada modelo listado na leaderboard
    for model_id in leaderboard_df["model_id"]:
        model = h2o.get_model(model_id)

        pred_h2o = model.predict(train_h2o)
        pred_df = pred_h2o.as_data_frame()

        # Coluna 'predict' costuma vir como '0' / '1' (string); converte pra int
        y_pred = pred_df["predict"].astype(int).values

        acc_list.append(accuracy_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        f1_list.append(f1_score(y_true, y_pred))

    # Adiciona as métricas como colunas
    leaderboard_df["accuracy"] = acc_list
    leaderboard_df["recall"] = rec_list
    leaderboard_df["f1"] = f1_list

    # Deixar só as colunas desejadas no CSV
    cols_to_keep = [
        "model_id",
        "training_time_ms",
        "predict_time_per_row_ms",
        "accuracy",
        "recall",
        "f1",
    ]
    cols_exist = [c for c in cols_to_keep if c in leaderboard_df.columns]
    leaderboard_df = leaderboard_df[cols_exist]

    leaderboard_df.to_csv(out_csv, index=False)

def ler_paciente_manual() -> pd.DataFrame:
        print("\n=== Digite os dados do novo paciente (H2O) ===")
        preg = int(input("Pregnancies (nº de gestações): "))
        glu = float(input("Glucose (glicose plasmática): "))
        bp = float(input("BloodPressure (pressão diastólica): "))
        skin = float(input("SkinThickness (espessura de pele): "))
        ins = float(input("Insulin (insulina sérica): "))
        bmi = float(input("BMI (índice de massa corporal): "))
        dpf = float(input("DiabetesPedigreeFunction: "))
        age = int(input("Age (idade): "))

        dados = {
            "Pregnancies": [preg],
            "Glucose": [glu],
            "BloodPressure": [bp],
            "SkinThickness": [skin],
            "Insulin": [ins],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age],
        }
        return pd.DataFrame(dados)

def prever_paciente_h2o(aml):
    # lê do teclado
    novo_pdf = ler_paciente_manual()
    # converte para H2OFrame
    novo_hf = h2o.H2OFrame(novo_pdf)

    pred = aml.leader.predict(novo_hf).as_data_frame()

    # colunas típicas: 'predict', 'p0', 'p1'
    classe = pred.loc[0, 'predict']
    prob_diab = pred.loc[0, 'p1']  # prob da classe 1 (diabético)

    print("\n=== Resultado H2O AutoML ===")
    if str(classe) == '1':
        print(f"-> O modelo prevê que o paciente TEM diabetes.")
    else:
        print(f"-> O modelo prevê que o paciente NÃO tem diabetes.")

    print(f"-> Probabilidade de diabetes (classe 1): {prob_diab:.2%}")


def main():

    h2o.init()
    train_pdf = read_csv(TRAIN_PATH)    # ground truth (Outcome)
    aml = treinar_modelo(train_pdf)
    salvar_leaderboard(aml, train_pdf, out_csv=LEADERBOARD_CSV)
    h2o.cluster().shutdown(prompt=False)
    prever_paciente_h2o(aml)


if __name__ == '__main__':
    main()

