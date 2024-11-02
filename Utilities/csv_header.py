import csv

# Define the headers for the CSV file
headers = ["ID", "Stance", "Phase", "Head", "Chest", "Knee", "Heel"]

# Add headers for the X, Y, Z coordinates for each of the 32 landmarks
for i in range(0, 33):  # 32 landmarks
    headers.append(f"X{i}")
    headers.append(f"Y{i}")
    headers.append(f"Z{i}")

# Write the headers to a CSV file
csv_path = r"/Datasets/kin.csv"  # Replace with your desired CSV file path

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

print(f"Headers written to {csv_path}")
