import json

# Load JSON dari file
with open("clusters.json", "r") as f:
    data = json.load(f)

# Konversi ke format heatmap.js
heatmap_data = [
    {"x": d["x"], "y": d["y"], "value": d["intensity"]}
    for d in data["clusters"]
]

# Simpan hasil konversi
with open("heatmap_ready.json", "w") as f:
    json.dump(heatmap_data, f, indent=4)

print("Konversi selesai! File tersimpan sebagai heatmap_ready.json")
