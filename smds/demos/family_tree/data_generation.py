import os
import random
import pandas as pd


def load_names():
    path = "smds/demos/resources/names.csv"
    if not os.path.exists(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/names.csv"))

    df = pd.read_csv(path)
    return df['name'].tolist()


def generate_family_tree_data(names, n_samples=50):
    data = []

    for _ in range(n_samples):
        if len(names) < 7:
            raise ValueError("Not enough names in the dataset to generate a depth-2 tree (needs 7 unique names).")

        family_names = random.sample(names, 7)

        child = family_names[0]
        p1, p2 = family_names[1], family_names[2]
        gp1, gp2 = family_names[3], family_names[4]
        gp3, gp4 = family_names[5], family_names[6]

        base_text = (
            f"{child}'s parents are {p1} and {p2}. "
            f"{p1}'s parents are {gp1} and {gp2}. "
            f"{p2}'s parents are {gp3} and {gp4}. "
            f"Therefore, the family's youngest member is {child}."
        )

        for i in range(7):
            target_name = family_names[i]

            text = base_text + f" The family member is {target_name}."

            if i == 0:
                dist = 0
            elif 1 <= i <= 2:
                dist = 1
            elif 3 <= i <= 6:
                dist = 2

            entry = {
                "text": text,
                "names": family_names,
                "target_map": {target_name: dist}
            }
            data.append(entry)

    return pd.DataFrame(data)
