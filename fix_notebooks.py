import nbformat, os, shutil

fixed = 0
for root, _, files in os.walk('.'):
    for f in files:
        if f.endswith('.ipynb'):
            path = os.path.join(root, f)
            # Make a backup copy first
            shutil.copy2(path, path + ".bak")

            nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
            widgets = nb.metadata.get("widgets")
            if widgets and "state" not in widgets:
                print(f"Fixing {path}")
                del nb.metadata["widgets"]
                nbformat.write(nb, path)
                fixed += 1

print(f"Done! Fixed {fixed} notebooks.")
print("Backup copies saved as .ipynb.bak")
