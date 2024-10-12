import argparse
import os
import re



directory = "exports/"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, help="Threshold for experiments to delete. All experiments below the threshold will be deleted.", default=-1)
    parser.add_argument("-p", type=str, help="Pattern to match for filenames. If none is given, 'ppo_conv_model_exp_' is used as default.", default="ppo_conv_model_exp_")

    args = parser.parse_args()
    pattern = re.compile(args.p + "(\d+)\.pth")
    threshold = args.t
    paths = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if threshold < 0 or num <= threshold:
                file_path = os.path.join(directory, filename)
                paths.append(file_path)
                # print(f"path to delete: {file_path}")
    if not paths:
        print("There are no files to delete.")
        quit()
    input_request = "\n".join([p for p in paths])
    res:str = input(f"You are about to delete the following files:\n{input_request}\nProceed[N/y]?")
    if res.lower() == "y" or res.lower() == "yes":
        for p in paths:
            os.remove(p)
    else:
        print("Cancelled deletion.")
