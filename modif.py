import os
import shutil

label_dir = "labels/val"  
backup_dir = label_dir + "_backup"

if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    for f in os.listdir(label_dir):
        if f.endswith(".txt"):
            shutil.copy(os.path.join(label_dir, f), os.path.join(backup_dir, f))
    print(f"🛡 Backup enregistrée dans : {backup_dir}")

class_map = {
    15: 0,
    16: 1,
    17: 2,
    18: 3,
    19: 4,
    20: 5,
    21: 6,
    22: 7,
    23: 8,
    24: 9
}

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts or len(parts) < 5:
                continue  

            old_class = int(parts[0])
            new_class = class_map.get(old_class, old_class)  
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

print("✅ Conversion des classes terminée avec succès.")
