import pandas as pd
if __name__ == "__main__":
    files = ["PRJNA306016", "PRJEB11694", "PRJEB26162", "PRJEB20721", "PRJEB28097", "PRJEB15485", "PRJEB6997"]
    for file_name in files:
        with open(file_name + ".txt", "r") as f:
            header = f.readline()
            header = header.split("\t")
            header[-1] = header[-1].strip("\n")

            content = f.readlines()
            content = [c.split("\t") for c in content]
            df = pd.DataFrame(content)
            df = df.replace('\n', '', regex=True)
            df.columns = header
            df.to_csv(file_name + ".csv")

    print(content[0])







