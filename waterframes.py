import sys

input_pdb = sys.argv[1]
output_pdb = input_pdb+"waters_as_frames.pdb"

with open(input_pdb, 'r') as f:
    lines = f.readlines()

# Filter: keep only lines starting with "ATOM"
atom_lines = [line for line in lines if line.startswith("ATOM")]

# Number of water molecules (each has 3 lines)
nwats = len(atom_lines) // 3

with open(output_pdb, 'w') as out:
    model = 1
    index = 0
    while index < len(atom_lines):
        out.write(f"MODEL     {model}\n")
        # Write 3 lines for this water
        for i in range(3):
            out.write(atom_lines[index + i])
        out.write("ENDMDL\n")
        index += 3
        model += 1
