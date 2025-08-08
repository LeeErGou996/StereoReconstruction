def read_off_with_color(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    if not lines or lines[0] not in ['OFF', 'COFF']:
        raise ValueError("Not a valid OFF or COFF file")

    header = lines[0]
    num_vertices, num_faces, _ = map(int, lines[1].split())

    verts = []
    for i in range(num_vertices):
        parts = lines[2 + i].split()
        x, y, z = map(float, parts[:3])
        if header == 'COFF' and len(parts) >= 6:
            r, g, b = map(int, parts[3:6])
        else:
            r, g, b = 255, 255, 255
        verts.append((x, y, z, r, g, b))

    return verts


def write_ply(filepath, verts):
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in verts:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


if __name__ == "__main__":
    off_file = "bunny_part2_trans.off"
    ply_file = "pointcloud_right.ply"
    verts = read_off_with_color(off_file)
    write_ply(ply_file, verts)
    print(f"✅ Converted {off_file} to {ply_file}")
