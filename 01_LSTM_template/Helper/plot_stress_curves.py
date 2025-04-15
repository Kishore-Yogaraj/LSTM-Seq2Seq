import os
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
folder_path = os.path.join(os.getcwd(), "Train_Group_Multi")
plot_save_path = os.path.join(os.getcwd(), "plots")
column_name = "sigma1eff[kN/m²]"

# === Check if data folder exists ===
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"Folder not found: {folder_path}")

# === Create plot output folder ===
os.makedirs(plot_save_path, exist_ok=True)
plot_filename = os.path.join(plot_save_path, "stress_curves.png")

# === Set up plot ===
plt.figure(figsize=(12, 6))
plt.title("Stress Curves for Soil Samples")
plt.xlabel("Time Step (Index)")
plt.ylabel("Stress (kN/m²)")

# === Iterate over each .txt file ===
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, sep='\t', header=0)
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in {filename}")
                continue

            stress_values = df[column_name].values
            plt.plot(stress_values, label=filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# === Finalize plot ===
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save plot ===
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")

# === Optional: Display plot (comment out if not needed) ===
# plt.show()
