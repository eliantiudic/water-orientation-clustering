#!/usr/bin/env python

import sys
import numpy as np
from sklearn.cluster import DBSCAN


# PARSING LOGIC
def parse_pdb_water_file(inputfile):
    """
    Reads 'inputfile' and returns lists: oxygen_lines, h1_lines, h2_lines.
    Each water is 3 lines: O, H1, H2.
    We assume lines[1:end:3], lines[2:end:3], lines[3:end:3] for O/H1/H2
    (matching your original code).
    """
    with open(inputfile, 'r') as f:
        lines = f.readlines()
    end = len(lines)
    oxygen = lines[1:end:3]  # lines 1,4,7,...
    h1     = lines[2:end:3]  # lines 2,5,8,...
    h2     = lines[3:end:3]  # lines 3,6,9,...
    return oxygen, h1, h2

def extract_coordinates(lines):
    """
    Extract x,y,z from columns [32:38], [40:46], [48:54] for each line.
    Returns numpy arrays x, y, z.
    """
    x_list, y_list, z_list = [], [], []
    for line in lines:
        x = float(line[32:38])
        y = float(line[40:46])
        z = float(line[48:54])
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return np.array(x_list), np.array(y_list), np.array(z_list)


# QUATERNION & CANONICAL ORIENTATION LOGIC
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def quaternion_from_vector_alignment(u, v):
    """
    Returns a quaternion rotating normalized u onto normalized v, each 3D.
    q = [q0, q1, q2, q3].
    """
    u = normalize_vector(u)
    v = normalize_vector(v)
    dot = np.dot(u, v)
    # parallel
    if abs(dot - 1.0) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    # opposite
    if abs(dot + 1.0) < 1e-12:
        perp = np.cross(u, np.array([1.,0.,0.], dtype=float))
        if np.linalg.norm(perp) < 1e-12:
            perp = np.cross(u, np.array([0.,1.,0.], dtype=float))
        perp = normalize_vector(perp)
        return np.array([0.0, perp[0], perp[1], perp[2]], dtype=float)
    # general case
    axis = np.cross(u, v)
    axis = normalize_vector(axis)
    angle = np.arccos(dot)
    half = angle / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)

def quaternion_multiply(q1, q2):
    """
    Hamilton product of two quaternions q1,q2 = [w,x,y,z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)

def apply_quaternion(q, vec):
    """
    Rotate 3D vector vec by quaternion q, returning the rotated vector.
    """
    qc = np.array([ q[0], -q[1], -q[2], -q[3] ])  # conjugate
    tmp = quaternion_multiply(q, np.array([0., vec[0], vec[1], vec[2]]))
    out = quaternion_multiply(tmp, qc)
    return out[1:]  # skip the w-component

def canonical_quaternion_for_water(h1_vec, h2_vec):
    """
    1) Sort (h1_vec, h2_vec) so h1 < h2 lex order => consistent labeling
    2) Rotate h1 -> x-axis
    3) Rotate about x-axis if needed so h2 goes into xy-plane
    4) Fix sign q0 >= 0
    """
    # step 1
    if tuple(h2_vec) < tuple(h1_vec):
        h1_vec, h2_vec = h2_vec, h1_vec

    # step 2
    x_ref = np.array([1., 0., 0.])
    q1 = quaternion_from_vector_alignment(h1_vec, x_ref)

    # step 3
    h2p = apply_quaternion(q1, h2_vec)
    if abs(h2p[2]) > 1e-12:
        angle = np.arctan2(h2p[2], h2p[1])
        half = angle / 2.0
        s = np.sin(half)
        rot_x = np.array([np.cos(half), s, 0., 0.])
        q2 = quaternion_multiply(rot_x, q1)
    else:
        q2 = q1

    # step 4: fix sign
    if q2[0] < 0:
        q2 = -q2
    return q2

# MAIN SCRIPT
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} inputfile.pdb")
        sys.exit(1)
    inputfile = sys.argv[1]

    # Output name based on input
    # e.g., "cluster.000001.pdb" -> "cluster.000001_dbscan_centroids.pdb"
    if inputfile.lower().endswith('.pdb'):
        base = inputfile[:-4]  # remove ".pdb"
    else:
        base = inputfile
    outfilename = f"{base}_dbscan_centroids.pdb"

    # 1) Parse the file
    oxy_lines, h1_lines, h2_lines = parse_pdb_water_file(inputfile)
    ox, oy, oz = extract_coordinates(oxy_lines)
    h1x, h1y, h1z = extract_coordinates(h1_lines)
    h2x, h2y, h2z = extract_coordinates(h2_lines)
    N = len(ox)
    if not (len(h1x) == N == len(h2x)):
        print("Error: mismatch in O/H1/H2 counts. Check file format.")
        sys.exit(2)

    # 2) Build canonical quaternions
    quaternions = np.zeros((N,4), dtype=float)
    for i in range(N):
        oh1 = np.array([h1x[i]-ox[i], h1y[i]-oy[i], h1z[i]-oz[i]])
        oh2 = np.array([h2x[i]-ox[i], h2y[i]-oy[i], h2z[i]-oz[i]])
        q_i = canonical_quaternion_for_water(oh1, oh2)
        quaternions[i] = q_i

    # 3) DBSCAN
    eps_value = 0.07  # tune as desired
    db = DBSCAN(eps=eps_value, min_samples=5, metric='euclidean', n_jobs=-1)
    db.fit(quaternions)
    labels = db.labels_

    unique_labels = sorted(set(labels))
    num_clusters = sum(1 for x in unique_labels if x != -1)

    # 4) Print to a file:
    with open(outfilename, "w") as f:
        # remarks
        f.write("REMARK  DBSCAN cluster results\n")
        f.write(f"REMARK  Total waters: {N}\n")
        f.write(f"REMARK  Eps cutoff: {eps_value}\n\n\n")

        for cluster_id in unique_labels:
            if cluster_id == -1:
                # skip noise
                continue

            idxs = np.where(labels == cluster_id)[0]
            size = len(idxs)
            percent = 100.0 * size / N
            # only print cluster if > 5%
            if percent < 5.0:
                continue

            # find centroid in quaternion space
            cluster_quats = quaternions[idxs, :]
            centroid = cluster_quats.mean(axis=0)
            # re-normalize in case we want a "unit quaternion" average
            cent_norm = np.linalg.norm(centroid)
            if cent_norm > 1e-12:
                centroid /= cent_norm

            # find the water i in idxs whose quaternion is closest to the centroid
            best_i = -1
            best_dist = 1.0e9
            for i in idxs:
                dist = np.linalg.norm(quaternions[i] - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            # Oxygen at origin
            Ox = 0.0
            Oy = 0.0
            Oz = 0.0

            # Translate H1 and H2 so the O is at (0,0,0)
            H1x = h1x[best_i] - ox[best_i]
            H1y = h1y[best_i] - oy[best_i]
            H1z = h1z[best_i] - oz[best_i]
            H2x = h2x[best_i] - ox[best_i]
            H2y = h2y[best_i] - oy[best_i]
            H2z = h2z[best_i] - oz[best_i]

            # Print the representative water, with O at origin
            f.write(f"REMARK  Cluster {cluster_id}: {size} waters ({percent:.2f}%)\n")

            f.write("ATOM  {:5d}  {:>2s}  SOL{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                best_i, "OW", cluster_id, Ox, Oy, Oz
            ))
            f.write("ATOM  {:5d}  {:>2s}  SOL{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                best_i, "H1", cluster_id, H1x, H1y, H1z
            ))
            f.write("ATOM  {:5d}  {:>2s}  SOL{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                best_i, "H2", cluster_id, H2x, H2y, H2z
            ))
            f.write("\n")
            '''
            # Now best_i is the "representative" for the cluster
            f.write(f"REMARK  Cluster {cluster_id}: {size} waters ({percent:.2f}%)\n")

            # Print only that one water's O/H1/H2
            # Use 'best_i' as atom number, or you can pick something else
            f.write("ATOM  {:5d}  {:>2s}  SOL{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                best_i, "OW", cluster_id, ox[best_i], oy[best_i], oz[best_i]
            ))
            f.write("ATOM  {:5d}  {:>2s}  SOL{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                best_i, "H1", cluster_id, h1x[best_i], h1y[best_i], h1z[best_i]
            ))
            f.write("ATOM  {:5d}  {:>2s}  SOL{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                best_i, "H2", cluster_id, h2x[best_i], h2y[best_i], h2z[best_i]
            ))
            f.write("\n")
            '''

    print(f"DBSCAN finished. Found {num_clusters} total clusters (some may be under 5%).")
    print(f"Wrote file: {outfilename}")

if __name__ == "__main__":
    main()
