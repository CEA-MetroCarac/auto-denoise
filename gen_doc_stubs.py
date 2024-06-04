"""
Automatic documentation generation.
"""

from pathlib import Path

import mkdocs_gen_files

src_root = Path("src/autoden")
for path in src_root.glob("**/*.py"):
    doc_path = Path("reference", path.relative_to(src_root)).with_suffix(".md")
    if path.parts[-1][:2] == "__":
        print(f"Excluding documentation page for: {doc_path}")
        continue
    print(f"Generating documentation page for: {doc_path}")

    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = ".".join(path.with_suffix("").parts[1:])
        print("::: " + ident, file=f)
