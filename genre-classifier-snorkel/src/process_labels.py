from load_data import load_plain_text_dfs

if __name__ == "__main__":
    df_train, _ = load_plain_text_dfs(language="english")
    print(df_train)