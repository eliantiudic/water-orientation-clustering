#!/usr/bin/env python3
import sys
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN

def read_waters_from_pdb(pdb_file):
    """
    Reads a PDB file where each water is described by three lines:
      - O  (e.g., 'OW')
      - H1 (e.g., 'H1')
      - H2 (e.g., 'H2')
    Returns a list of NumPy arrays of shape (3, 3):
        [
          array([[Ox, Oy, Oz],
                 [H1x, H1y, H1z],
                 [H2x, H2y, H2z]]),
        ]
    """
    waters = []
    temp_coords = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                fields = line.split()
                x = float(fields[5])
                y = float(fields[6])
                z = float(fields[7])
                temp_coords.append((x, y, z))

                # Once we have 3 lines => 1 water
                if len(temp_coords) == 3:
                    water_array = np.array(temp_coords, dtype=np.float64)  # shape (3,3)
                    waters.append(water_array)
                    temp_coords = []

    return waters

def cluster_orientations(waters, eps=0.15, min_samples=5):
    """
    Clusters water orientations using DBSCAN, omitting the oxygen coordinates from the descriptor.
      1) Translate oxygen to the origin (O -> (0,0,0), H1 -> H1 - O, H2 -> H2 - O).
      2) Normalize the resulting hydrogen vectors so each has length 1.
      3) Flatten just the hydrogen coordinates into a 6D vector:
         [H1x, H1y, H1z, H2x, H2y, H2z].
    """
    descriptors = []

    for water in waters:
        # water[0] = [Ox, Oy, Oz]
        # water[1] = [H1x, H1y, H1z]
        # water[2] = [H2x, H2y, H2z]
        O = water[0]
        H1 = water[1]
        H2 = water[2]

        # 1) Translate
        H1_trans = H1 - O
        H2_trans = H2 - O

        # 2) Normalize hydrogen vectors
        norm_H1 = np.linalg.norm(H1_trans)
        if norm_H1 > 1e-12:
            H1_trans = H1_trans / norm_H1

        norm_H2 = np.linalg.norm(H2_trans)
        if norm_H2 > 1e-12:
            H2_trans = H2_trans / norm_H2

        # 3) Flatten into 6D (omit oxygen)
        descriptor = np.concatenate((H1_trans, H2_trans))
        descriptors.append(descriptor)

    descriptors = np.array(descriptors)  # shape (n_waters, 6)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(descriptors)
    return labels

def main():
    if len(sys.argv) < 2:
        print("Usage: python cluster_water_orientations.py <pdb_file> [eps] [min_samples] [pdb_output]")
        sys.exit(1)

    pdb_file = sys.argv[1]
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    min_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    pdb_output = sys.argv[4] if len(sys.argv) > 4 else "orientational_clusters_of_{}".format(pdb_file[0:18])

    # 1) Parse the PDB to get waters 
    waters = read_waters_from_pdb(pdb_file)
    if not waters:
        print(f"No water data found in '{pdb_file}'. Check format or file content.")
        sys.exit(0)

    # 2) Cluster with DBSCAN
    labels = cluster_orientations(waters, eps=eps, min_samples=min_samples)

    # 3) Analyze results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    label_counts = Counter(labels)
    total = len(labels)

    # 4) Identify a representative for each cluster
    reps = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        if lab not in reps:
            reps[lab] = i  # store the first index in the cluster as representative

    # Write output to pdb file
    with open(pdb_output, 'w') as f:
        f.write("REMARKS\n")
        f.write("This pdb file has the most probable orientations in a water cluster.\n")
        f.write("We consider an orientational cluster if it has at least 10% of water population as neighbors.\n\n")

        atom_index = 1

        # Go through each cluster (except noise)
        for lab in sorted(reps.keys()):
            count = label_counts[lab]
            pct = 100.0 * count / total

            # Only write to PDB if cluster is >= 10%
            if pct < 10.0:
                continue

            # Representative water for this cluster
            rep_index = reps[lab]
            O_xyz = waters[rep_index][0]
            H1_xyz = waters[rep_index][1]
            H2_xyz = waters[rep_index][2]

            # Oxygen (with cluster population)
            f.write(
                f"ATOM  {atom_index:5d}  OW  SOL C{lab:4d}    "
                f"{O_xyz[0]:8.3f}{O_xyz[1]:8.3f}{O_xyz[2]:8.3f}"
                f" {pct:7.3f}%\n"
            )
            atom_index += 1

            # Hydrogen 1
            f.write(
                f"ATOM  {atom_index:5d}  H1  SOL X{lab:4d}    "
                f"{H1_xyz[0]:8.3f}{H1_xyz[1]:8.3f}{H1_xyz[2]:8.3f}\n"
            )
            atom_index += 1

            # Hydrogen 2
            f.write(
                f"ATOM  {atom_index:5d}  H2  SOL X{lab:4d}    "
                f"{H2_xyz[0]:8.3f}{H2_xyz[1]:8.3f}{H2_xyz[2]:8.3f}\n"
            )
            atom_index += 1

            f.write("\n")  # Blank line after each cluster

        f.write("END\n")

    print(f"DBSCAN found {n_clusters} clusters (excluding noise).")
    print(f"Output PDB file written to '{pdb_output}'.")

if __name__ == "__main__":
    main()
