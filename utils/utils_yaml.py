import os

# root_dir = os.path.join(r"""Сварные швы""", r"""dataset""")
# classes_file = os.path.join(root_dir, "classes.txt")

def get_img_folder(root_folder: str) -> list:
    imgs_lst = list()
    anno_lst = list()
    for file in os.listdir(root_folder):
        if file.endswith(".jpg"):
            imgs_lst.append(os.path.join(root_folder, file))
        elif file.endswith(".txt"):
            anno_lst.append(os.path.join(root_folder, file))
        else:
            pass
    return imgs_lst, anno_lst

def sep_ds(root_folder: str, train_percent: int = 80) -> dict:
    root_lst, anno_lst = get_img_folder(root_folder)
    root_f_len = len(root_lst)
    train_len = int(root_f_len*train_percent*0.01)

    return {"train": root_lst[:train_len], "val": root_lst[train_len:], "anno_gen": anno_lst}

def generate_yaml(ds_root_dir: str, classes_file: str, yaml_name: str) -> None:

    ds = sep_ds(ds_root_dir)

    train = "train: [{}]\n".format(', '.join(ds["train"]))
    val = "val: [{}]\n".format(', '.join(ds["val"]))

    labels = "labels:\n\tdetection: [{}]\n".format(', '.join(ds["anno_gen"]))

    names = "names:\n"
    with open(classes_file, 'r', encoding='utf8') as f:
        for n, line in enumerate(f):
            names += "\t{}: {}".format(n, line)
    f.close()

    with open(yaml_name+'.yaml', 'w', encoding='utf8') as f:
        f.write(train + val + labels + names)
    f.close()