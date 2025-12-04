#!/usr/bin/env python3
"""
Generate PNG and SVG images from Mermaid diagrams using mermaid.ink API.

This script extracts all Mermaid diagrams from ARCHITECTURE_DIAGRAMS.md
and generates both PNG and SVG versions using the public mermaid.ink service.
"""

import re
import base64
import requests
from pathlib import Path
import zlib


def extract_diagrams(md_file: str) -> list[tuple[str, str]]:
    """Extract diagram names and content from markdown file."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match diagram sections
    pattern = r'## (\d+\.\s+.+?)\n\n```mermaid\n(.+?)```'
    matches = re.findall(pattern, content, re.DOTALL)

    diagrams = []
    for title, diagram_code in matches:
        # Clean title for filename
        clean_title = re.sub(r'^\d+\.\s+', '', title)  # Remove number prefix
        clean_title = re.sub(r'[^\w\s-]', '', clean_title)  # Remove special chars
        clean_title = clean_title.strip().replace(' ', '_')
        diagrams.append((clean_title, diagram_code.strip()))

    return diagrams


def encode_mermaid(diagram: str) -> str:
    """Encode Mermaid diagram for mermaid.ink API."""
    # Use pako (zlib) compression and base64 encoding
    compressed = zlib.compress(diagram.encode('utf-8'), level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
    return encoded


def generate_images(diagrams: list[tuple[str, str]], output_dir: str = 'diagrams'):
    """Generate PNG and SVG images from diagrams."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"📊 Generating diagram images in: {output_path}/")
    print("=" * 60)

    for i, (name, code) in enumerate(diagrams, 1):
        print(f"\n{i}. {name}")

        # Encode diagram
        encoded = encode_mermaid(code)

        # Generate PNG
        png_url = f"https://mermaid.ink/img/pako:{encoded}"
        png_file = output_path / f"{i:02d}_{name}.png"

        try:
            print(f"   Downloading PNG... ", end='', flush=True)
            png_response = requests.get(png_url, timeout=30)
            png_response.raise_for_status()

            with open(png_file, 'wb') as f:
                f.write(png_response.content)
            print(f"✅ Saved: {png_file}")
        except Exception as e:
            print(f"❌ Failed: {e}")

        # Generate SVG
        svg_url = f"https://mermaid.ink/svg/pako:{encoded}"
        svg_file = output_path / f"{i:02d}_{name}.svg"

        try:
            print(f"   Downloading SVG... ", end='', flush=True)
            svg_response = requests.get(svg_url, timeout=30)
            svg_response.raise_for_status()

            with open(svg_file, 'wb') as f:
                f.write(svg_response.content)
            print(f"✅ Saved: {svg_file}")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n" + "=" * 60)
    print(f"✅ Done! Generated {len(diagrams) * 2} files in {output_path}/")
    print("\nFiles created:")
    for file in sorted(output_path.glob('*')):
        print(f"  - {file.name}")


def main():
    """Main execution."""
    md_file = 'ARCHITECTURE_DIAGRAMS.md'

    if not Path(md_file).exists():
        print(f"❌ Error: {md_file} not found!")
        print("Make sure you're in the project root directory.")
        return 1

    print("🎨 Mermaid Diagram Generator")
    print("=" * 60)
    print(f"Source: {md_file}")
    print(f"Using: mermaid.ink API (no installation needed!)")
    print()

    # Extract diagrams
    print("Extracting diagrams from markdown...")
    diagrams = extract_diagrams(md_file)
    print(f"✅ Found {len(diagrams)} diagrams")

    # Generate images
    generate_images(diagrams)

    print("\n💡 Tip: Open PNG files in any image viewer")
    print("    or import SVG files into presentations/docs")

    return 0


if __name__ == '__main__':
    exit(main())
