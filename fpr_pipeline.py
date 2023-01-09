from fpr_utils import *

def main():
    print("Reading data...")
    df_raw = read_data(excel_path)
    print("Dataset preparation...")
    df = prepare_data(df_raw)
    handle_missings(df)
    replace_with_thresholds(df)
    df_features = feature_eng(df)
    df_encoded = encode(df_features)
    print("Modelling...")
    df_model, optimal_cluster_num= hyperparameter_optimization(df_encoded)
    print(f"**** Optimal number of clusters according to elbow method: {optimal_cluster_num} ****", end="\n\n")
    clusters = modelling(df_model, elbow_val = 20) # --> Cluster sayısını teknik bilgi ile biz belirledik.
    df_final = final_df(clusters)
    recommend_player(df_final, "Phil Foden", "Piyasa Degeri(euro)", 5)


if __name__ == "__main__":
    main()
