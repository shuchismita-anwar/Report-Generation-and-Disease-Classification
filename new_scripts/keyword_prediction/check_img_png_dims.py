import glob
import os
from PIL import Image

def check_image_dimensions():
    """Check actual image dimensions from PNG files in data_dump/output/img_png"""
    print("🔍 Checking PNG image dimensions in data_dump/output/img_png...")
    
    # Path to the PNG files directory (relative to current working directory)
    img_pattern = "../../data_dump/output/img_png/*.png"
    png_files = glob.glob(img_pattern)
    
    print(f"Found {len(png_files)} PNG files")
    
    if not png_files:
        print("❌ No PNG files found! Checking alternate path...")
        # Try absolute path from script location
        img_pattern_alt = "./data_dump/output/img_png/*.png"
        png_files = glob.glob(img_pattern_alt)
        print(f"Found {len(png_files)} PNG files with alternate path")
    
    dimensions = {}
    dimension_counts = {}
    
    # Check PNG files
    if png_files:
        print(f"\n📊 Checking PNG file dimensions...")
        for i, file_path in enumerate(png_files):
            try:
                with Image.open(file_path) as img:
                    dims = img.size  # (width, height)
                    filename = os.path.basename(file_path)
                    dimensions[filename] = dims
                    
                    # Count occurrences of each dimension
                    if dims in dimension_counts:
                        dimension_counts[dims] += 1
                    else:
                        dimension_counts[dims] = 1
                    
                    # Print first 10 files for reference
                    if i < 10:
                        print(f"  {filename}: {dims[0]}x{dims[1]}")
                    elif i == 10:
                        print(f"  ... (showing first 10 files)")
                        
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
    else:
        print("❌ No PNG files found! Check the path structure.")
        print("Expected structure: data_dump/output/img_png/*.png")
    
    print(f"\n📈 Summary:")
    print(f"Total files processed: {len(dimensions)}")
    print(f"Unique image dimensions found: {len(dimension_counts)}")
    
    for dims, count in sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(dimensions)) * 100 if dimensions else 0
        print(f"  {dims[0]}x{dims[1]} pixels: {count} files ({percentage:.1f}%)")
    
    # Find min and max dimensions
    if dimensions:
        all_dims = list(dimensions.values())
        widths = [d[0] for d in all_dims]
        heights = [d[1] for d in all_dims]
        
        print(f"\n📏 Dimension Statistics:")
        print(f"Width  - Min: {min(widths)}, Max: {max(widths)}")
        print(f"Height - Min: {min(heights)}, Max: {max(heights)}")
        
        # Check if all images have the same dimensions
        if len(dimension_counts) == 1:
            dims = list(dimension_counts.keys())[0]
            print(f"✅ All images have consistent dimensions: {dims[0]}x{dims[1]}")
        else:
            print(f"⚠️  Images have varying dimensions!")
    
    return dimensions, dimension_counts

if __name__ == "__main__":
    dims, counts = check_image_dimensions() 