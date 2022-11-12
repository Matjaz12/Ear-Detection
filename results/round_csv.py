import pandas as pd

if __name__ == "__main__":
    FILE_PATH = "./vj_left_right_ear_cascade_GRAY_2"
    df = pd.read_csv(FILE_PATH + ".csv") 
    df = df.round(3)

    df.to_csv(FILE_PATH + ".csv", index=False)
    print(df)