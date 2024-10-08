import os

# execution outcomes
trn_dir = "../../../data/output/xxxxxx"

dataset_dirs = ["DeepMDatasets", "Wikidata", "assays", "miller2", "prospect"]
# dataset_dirs = ['assays', 'miller2', 'prospect']
dataset_name_ls = []
for dataset in dataset_dirs:
    directory = "../../data/sm-valentine-m1/{}".format(dataset)
    for subdir in os.listdir(directory):
        if subdir in [
            "Joinable",
            "Semantically-Joinable",
            "Unionable",
            "View-Unionable",
        ]:
            subdir = os.path.join(directory, subdir)
            for dir in os.listdir(subdir):
                for file in os.listdir(os.path.join(subdir, dir)):
                    abs_file_path = os.path.join(subdir, dir, file)
                    if abs_file_path.endswith("source.csv"):
                        orig_file_src = abs_file_path
                    elif abs_file_path.endswith("target.csv"):
                        orig_file_tgt = abs_file_path
                    elif abs_file_path.endswith("mapping.json"):
                        orig_file_golden_matches = abs_file_path
                cate_dir = "/".join(subdir.split("/")[-2:])
                dataset_name = f"{cate_dir}/{dir}"
                dataset_name_ls.append(dataset_name)

        elif subdir in ["Musicians"]:
            subdir = os.path.join(directory, subdir)
            for dir in os.listdir(subdir):
                for file in os.listdir(os.path.join(subdir, dir)):
                    abs_file_path = os.path.join(subdir, dir, file)
                    if abs_file_path.endswith("source.csv"):
                        orig_file_src = abs_file_path
                    elif abs_file_path.endswith("target.csv"):
                        orig_file_tgt = abs_file_path
                    elif abs_file_path.endswith("mapping.json"):
                        orig_file_golden_matches = abs_file_path
                cate_dir = "/".join(subdir.split("/")[-2:])
                dataset_name = f"{cate_dir}/{dir}"
                dataset_name_ls.append(dataset_name)

        elif os.path.isdir(os.path.join(directory, subdir)):
            for file in os.listdir(os.path.join(directory, subdir)):
                abs_file_path = os.path.join(directory, subdir, file)
                if abs_file_path.endswith("source.csv"):
                    orig_file_src = abs_file_path
                elif abs_file_path.endswith("target.csv"):
                    orig_file_tgt = abs_file_path
                elif abs_file_path.endswith("mapping.json"):
                    orig_file_golden_matches = abs_file_path
            cate_dir = directory.split("/")[-1]
            dataset_name = f"{cate_dir}/{subdir}"
            dataset_name_ls.append(dataset_name)

print("total numbers of datasets {}".format(len(dataset_name_ls)))
# print(dataset_name_ls[0])
# print(dataset_name_ls[-1])

dataset_name_ls = [element.replace("/", "-") for element in dataset_name_ls]


curr_files = []
for root, _, files in os.walk(trn_dir):
    for file in files:
        if file.endswith(".json"):
            curr_files.append(file)
print(len(curr_files))

trn_files = []
dup_trn_files = []
for f in curr_files:
    # curr_f = f[20:-5]
    curr_f = f[:-5]
    if curr_f not in trn_files:
        trn_files.append(curr_f)
    else:
        dup_trn_files.append(curr_f)

not_trn_files = []
for data in dataset_name_ls:
    if data not in trn_files:
        not_trn_files.append(data)
print(len(not_trn_files))

print("has trained: ", len(trn_files))
print("dup trn data: ", dup_trn_files)
print("lack trn data: ", not_trn_files)

if len(dup_trn_files) > 0:
    while True:
        user_input = input("do you want to remove the duplicated trn data: (yes or no)")
        if user_input.lower() in ("yes", "no"):
            break
        else:
            print("Invalid input. Please try again.")
    if user_input.lower() == "yes":
        print("begin to remove the duplicated trn data:")
        for f in curr_files:
            for dup_f in dup_trn_files:
                if dup_f in f:
                    os.remove(os.path.join(trn_dir, f))
                    dup_trn_files.remove(dup_f)
    else:
        print("the duplicated trn data will not be removed:")

s = set()
for f in not_trn_files:
    s.add(f.split("_", 1)[0])
print("lack data's category")
print(s)
