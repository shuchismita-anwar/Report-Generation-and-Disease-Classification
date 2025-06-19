import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import json
from PIL import Image
from reflacx_region import REFLACXRegionAnalyzer

class AnatomicalRegionAverager:
    def __init__(self, base_path="../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye"):
        self.base_path = base_path
        self.expected_regions = [
            'cardiac silhouette',
            'left clavicle', 
            'left costophrenic angle',
            'left hilar structures',
            'left lower lung zone',
            'left lung',
            'left mid lung zone', 
            'left upper lung zone',
            'right clavicle',
            'right costophrenic angle', 
            'right hilar structures',
            'right lower lung zone',
            'right lung',
            'right mid lung zone',
            'right upper lung zone',
            'trachea',
            'upper mediastinum'
        ]
        
        # Store all bounding box data for averaging with image dimensions
        self.region_data = {region: [] for region in self.expected_regions}
        self.image_dimensions = {}  # Cache for image dimensions
        
    def get_image_dimensions(self, dicom_id):
        """Get dimensions for a specific image, with caching"""
        if dicom_id in self.image_dimensions:
            return self.image_dimensions[dicom_id]
        
        # Use the correct path pattern from check_image_dims.py
        # Images are in: mimic-eye/patient_*/CXR-JPG/s*/*.jpg
        image_patterns = [
            f"{self.base_path}/patient_*/CXR-JPG/s*/{dicom_id}.jpg",
            f"{self.base_path}/patient_*/CXR-JPG/*/{dicom_id}.jpg",
            f"{self.base_path}/*/CXR-JPG/s*/{dicom_id}.jpg"
        ]
        
        for pattern in image_patterns:
            try:
                image_files = glob.glob(pattern)
                if image_files:
                    try:
                        with Image.open(image_files[0]) as img:
                            width, height = img.size
                            self.image_dimensions[dicom_id] = (width, height)
                            return (width, height)
                    except Exception as e:
                        print(f"Error reading image {image_files[0]}: {e}")
                        continue
            except Exception as e:
                # Handle glob.glob errors (like path too long on Windows)
                print(f"Error globbing pattern {pattern}: {e}")
                continue
        
        # If no image found, use most common dimension as fallback
        if not hasattr(self, '_fallback_warned'):
            print(f"⚠️  Image not found for {dicom_id}, using fallback dimensions")
            self._fallback_warned = True
        fallback_dims = (2544, 3056)  # Most common from your analysis
        self.image_dimensions[dicom_id] = fallback_dims
        return fallback_dims
    
    def extract_dicom_id_from_path(self, file_path):
        """Extract DICOM ID from file path"""
        # For EyeGaze: bounding_boxes_{dicom_id}.csv
        # For REFLACX: {dicom_id}_bounding_boxes.csv
        basename = os.path.basename(file_path)
        
        if basename.startswith('bounding_boxes_'):
            # EyeGaze format
            dicom_id = basename.replace('bounding_boxes_', '').replace('.csv', '')
        elif basename.endswith('_bounding_boxes.csv'):
            # REFLACX format
            dicom_id = basename.replace('_bounding_boxes.csv', '')
        elif '_bboxes.csv' in basename:
            # Alternative REFLACX format: {dicom_id}_bboxes.csv
            dicom_id = basename.replace('_bboxes.csv', '')
        else:
            # Try to extract from middle
            parts = basename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                dicom_id = '_'.join(parts[:-2]) if 'bounding' in basename else '_'.join(parts)
            else:
                dicom_id = basename.replace('.csv', '')
        
        # Debug print for first few extractions
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 5:
            print(f"  🔍 Debug: {basename} -> {dicom_id}")
            self._debug_count += 1
        
        return dicom_id
    
    def find_eyegaze_bbox_files(self):
        """Find all EyeGaze bounding box CSV files"""
        pattern = f"{self.base_path}/*/EyeGaze/bounding_boxes_*.csv"
        bbox_files = glob.glob(pattern)
        print(f"Found {len(bbox_files)} EyeGaze bounding box files")
        return bbox_files
    
    def get_complete_reflacx_files(self):
        """Get list of REFLACX files with all 17 regions"""
        print("🔍 Analyzing REFLACX files to find complete ones...")
        analyzer = REFLACXRegionAnalyzer(self.base_path)
        results = analyzer.analyze_all_files()
        complete_files = analyzer.get_complete_files_list(results)
        print(f"Found {len(complete_files)} complete REFLACX files")
        return complete_files
    
    def process_eyegaze_file(self, file_path):
        """Process a single EyeGaze bounding box file"""
        try:
            df = pd.read_csv(file_path)
            dicom_id = self.extract_dicom_id_from_path(file_path)
            img_width, img_height = self.get_image_dimensions(dicom_id)
            
            for _, row in df.iterrows():
                region_name = row['bbox_name']
                if region_name in self.expected_regions:
                    # EyeGaze format: x1, x2, y1, y2
                    bbox = {
                        'x1': row['x1'],
                        'y1': row['y1'], 
                        'x2': row['x2'],
                        'y2': row['y2'],
                        'img_width': img_width,
                        'img_height': img_height,
                        'source': 'EyeGaze',
                        'file': os.path.basename(file_path),
                        'dicom_id': dicom_id
                    }
                    self.region_data[region_name].append(bbox)
            
            return True
            
        except Exception as e:
            print(f"Error processing EyeGaze file {file_path}: {e}")
            return False
    
    def process_reflacx_file(self, file_path):
        """Process a single REFLACX bounding box file"""
        try:
            df = pd.read_csv(file_path)
            dicom_id = self.extract_dicom_id_from_path(file_path)
            img_width, img_height = self.get_image_dimensions(dicom_id)
            
            for _, row in df.iterrows():
                region_name = row['class_name']
                if region_name in self.expected_regions:
                    # REFLACX format: x1, y1, x2, y2
                    bbox = {
                        'x1': row['x1'],
                        'y1': row['y1'],
                        'x2': row['x2'], 
                        'y2': row['y2'],
                        'img_width': img_width,
                        'img_height': img_height,
                        'source': 'REFLACX',
                        'confidence': row['confidence'],
                        'file': os.path.basename(file_path),
                        'dicom_id': dicom_id
                    }
                    self.region_data[region_name].append(bbox)
            
            return True
            
        except Exception as e:
            print(f"Error processing REFLACX file {file_path}: {e}")
            return False
    
    def process_all_files(self):
        """Process all EyeGaze and complete REFLACX files"""
        print("🚀 Starting anatomical region averaging process...")
        
        # Process EyeGaze files
        print("\n📁 Processing EyeGaze files...")
        eyegaze_files = self.find_eyegaze_bbox_files()
        eyegaze_success = 0
        
        for i, file_path in enumerate(eyegaze_files):
            if i % 100 == 0:
                print(f"  EyeGaze progress: {i}/{len(eyegaze_files)}")
            
            if self.process_eyegaze_file(file_path):
                eyegaze_success += 1
        
        print(f"✅ Processed {eyegaze_success}/{len(eyegaze_files)} EyeGaze files")
        
        # Process complete REFLACX files
        print("\n📁 Processing complete REFLACX files...")
        reflacx_files = self.get_complete_reflacx_files()
        reflacx_success = 0
        
        for i, file_path in enumerate(reflacx_files):
            if i % 100 == 0:
                print(f"  REFLACX progress: {i}/{len(reflacx_files)}")
            
            if self.process_reflacx_file(file_path):
                reflacx_success += 1
        
        print(f"✅ Processed {reflacx_success}/{len(reflacx_files)} REFLACX files")
        
        return {
            'eyegaze_files_processed': eyegaze_success,
            'reflacx_files_processed': reflacx_success,
            'total_files_processed': eyegaze_success + reflacx_success
        }
    
    def calculate_averaged_regions(self):
        """Calculate averaged bounding boxes for each anatomical region using robust methods"""
        print("\n📊 Calculating averaged anatomical regions (normalized by image dimensions)...")
        
        averaged_regions = {}
        
        for region_name in self.expected_regions:
            bboxes = self.region_data[region_name]
            
            if not bboxes:
                print(f"⚠️  No data found for region: {region_name}")
                continue
            
            # Normalize coordinates to fractions first, then average
            normalized_bboxes = []
            
            for bbox in bboxes:
                # Normalize coordinates by image dimensions
                x1_norm = bbox['x1'] / bbox['img_width']
                y1_norm = bbox['y1'] / bbox['img_height']
                x2_norm = bbox['x2'] / bbox['img_width']
                y2_norm = bbox['y2'] / bbox['img_height']
                
                # Validate normalized coordinates
                if (0 <= x1_norm < x2_norm <= 1 and 
                    0 <= y1_norm < y2_norm <= 1):
                    normalized_bboxes.append({
                        'x1_norm': x1_norm,
                        'y1_norm': y1_norm,
                        'x2_norm': x2_norm,
                        'y2_norm': y2_norm,
                        'confidence': bbox.get('confidence', 1.0),
                        'source': bbox['source'],
                        'img_dimensions': (bbox['img_width'], bbox['img_height'])
                    })
                else:
                    print(f"    ⚠️  Filtering invalid normalized bbox in {region_name}: "
                          f"({x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}) "
                          f"from {bbox['img_width']}x{bbox['img_height']} image")
            
            if not normalized_bboxes:
                print(f"⚠️  No valid normalized bounding boxes for region: {region_name}")
                continue
            
            # Extract normalized coordinates and weights
            x1_values = [bbox['x1_norm'] for bbox in normalized_bboxes]
            y1_values = [bbox['y1_norm'] for bbox in normalized_bboxes]
            x2_values = [bbox['x2_norm'] for bbox in normalized_bboxes]
            y2_values = [bbox['y2_norm'] for bbox in normalized_bboxes]
            weights = [bbox['confidence'] for bbox in normalized_bboxes]
            
            # Calculate robust averages using both median and weighted average
            median_x1 = np.median(x1_values)
            median_y1 = np.median(y1_values)
            median_x2 = np.median(x2_values)
            median_y2 = np.median(y2_values)
            
            # Calculate confidence-weighted average
            weighted_x1 = np.average(x1_values, weights=weights)
            weighted_y1 = np.average(y1_values, weights=weights)
            weighted_x2 = np.average(x2_values, weights=weights)
            weighted_y2 = np.average(y2_values, weights=weights)
            
            # Use median as primary method (more robust to outliers)
            avg_x1 = median_x1
            avg_y1 = median_y1
            avg_x2 = median_x2
            avg_y2 = median_y2
            
            # Validate final normalized bounding box
            if avg_x1 >= avg_x2 or avg_y1 >= avg_y2:
                print(f"⚠️  Final normalized bbox invalid for {region_name}, using weighted average as fallback")
                avg_x1 = weighted_x1
                avg_y1 = weighted_y1
                avg_x2 = weighted_x2
                avg_y2 = weighted_y2
            
            # Collect image dimension statistics
            image_dims = [bbox['img_dimensions'] for bbox in normalized_bboxes]
            unique_dims = list(set(image_dims))
            dim_counts = {dim: image_dims.count(dim) for dim in unique_dims}
            
            # Calculate comprehensive statistics
            stats = {
                'count': len(bboxes),
                'valid_count': len(normalized_bboxes),
                'filtered_count': len(bboxes) - len(normalized_bboxes),
                'eyegaze_count': len([b for b in normalized_bboxes if b['source'] == 'EyeGaze']),
                'reflacx_count': len([b for b in normalized_bboxes if b['source'] == 'REFLACX']),
                'confidence_stats': {
                    'mean_confidence': np.mean(weights),
                    'min_confidence': min(weights),
                    'max_confidence': max(weights)
                },
                'image_dimension_stats': {
                    'unique_dimensions': unique_dims,
                    'dimension_counts': dim_counts,
                    'most_common_dimension': max(dim_counts.items(), key=lambda x: x[1])[0]
                },
                'normalized_coordinate_stats': {
                    'x1': {'median': median_x1, 'weighted_avg': weighted_x1, 'std': np.std(x1_values), 'min': min(x1_values), 'max': max(x1_values)},
                    'y1': {'median': median_y1, 'weighted_avg': weighted_y1, 'std': np.std(y1_values), 'min': min(y1_values), 'max': max(y1_values)},
                    'x2': {'median': median_x2, 'weighted_avg': weighted_x2, 'std': np.std(x2_values), 'min': min(x2_values), 'max': max(x2_values)},
                    'y2': {'median': median_y2, 'weighted_avg': weighted_y2, 'std': np.std(y2_values), 'min': min(y2_values), 'max': max(y2_values)}
                }
            }
            
            averaged_regions[region_name] = {
                'normalized_bbox': {
                    'x1': avg_x1,
                    'y1': avg_y1,
                    'x2': avg_x2,
                    'y2': avg_y2
                },
                'statistics': stats
            }
            
            filtered_msg = f" ({stats['filtered_count']} filtered)" if stats['filtered_count'] > 0 else ""
            dims_msg = f"({len(unique_dims)} different image sizes)"
            print(f"  ✅ {region_name}: {stats['valid_count']}/{stats['count']} valid samples{filtered_msg} "
                  f"(EyeGaze: {stats['eyegaze_count']}, REFLACX: {stats['reflacx_count']}) {dims_msg}")
        
        return averaged_regions
    
    def normalize_to_fractions(self, averaged_regions):
        """The regions are already normalized, so this just formats them properly"""
        print(f"\n🔄 Regions are already normalized by actual image dimensions...")
        
        normalized_regions = {}
        
        for region_name, data in averaged_regions.items():
            bbox = data['normalized_bbox']
            
            # Ensure values are between 0 and 1 (should already be)
            normalized_bbox = {k: max(0, min(1, v)) for k, v in bbox.items()}
            
            normalized_regions[region_name] = {
                'name': region_name,
                'bounds': (normalized_bbox['x1'], normalized_bbox['y1'], 
                          normalized_bbox['x2'], normalized_bbox['y2']),
                'normalized_bbox': normalized_bbox,
                'statistics': data['statistics']
            }
        
        return normalized_regions
    
    def generate_anatomical_regions_dict(self, normalized_regions):
        """Generate the ANATOMICAL_REGIONS dictionary for the main pipeline"""
        print("\n🔧 Generating ANATOMICAL_REGIONS dictionary...")
        
        # Create region key mapping (convert to valid Python identifiers)
        region_key_mapping = {
            'cardiac silhouette': 'cardiac_silhouette',
            'left clavicle': 'left_clavicle',
            'left costophrenic angle': 'left_costophrenic_angle',
            'left hilar structures': 'left_hilar_structures',
            'left lower lung zone': 'left_lower_lung_zone',
            'left lung': 'left_lung',
            'left mid lung zone': 'left_mid_lung_zone',
            'left upper lung zone': 'left_upper_lung_zone',
            'right clavicle': 'right_clavicle',
            'right costophrenic angle': 'right_costophrenic_angle',
            'right hilar structures': 'right_hilar_structures',
            'right lower lung zone': 'right_lower_lung_zone',
            'right lung': 'right_lung',
            'right mid lung zone': 'right_mid_lung_zone',
            'right upper lung zone': 'right_upper_lung_zone',
            'trachea': 'trachea',
            'upper mediastinum': 'upper_mediastinum'
        }
        
        anatomical_regions = {}
        
        for region_name, data in normalized_regions.items():
            key = region_key_mapping.get(region_name, region_name.replace(' ', '_'))
            
            # Generate unique keywords
            keywords = [region_name.lower()]
            key_as_text = key.replace('_', ' ')
            if key_as_text.lower() != region_name.lower():
                keywords.append(key_as_text.lower())
            
            anatomical_regions[key] = {
                'name': region_name,
                'keywords': keywords,
                'bounds': data['bounds'],
                'statistics': {
                    'sample_count': data['statistics']['valid_count'],
                    'total_count': data['statistics']['count'],
                    'filtered_count': data['statistics']['filtered_count'],
                    'eyegaze_samples': data['statistics']['eyegaze_count'],
                    'reflacx_samples': data['statistics']['reflacx_count'],
                    'mean_confidence': data['statistics']['confidence_stats']['mean_confidence']
                }
            }
        
        return anatomical_regions
    
    def save_results(self, normalized_regions, anatomical_regions, output_dir="anatomical_results"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, "detailed_anatomical_regions.json")
        with open(detailed_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_data = {}
            for region, data in normalized_regions.items():
                serializable_data[region] = {
                    'name': data['name'],
                    'bounds': data['bounds'],
                    'normalized_bbox': {k: float(v) for k, v in data['normalized_bbox'].items()},
                    'statistics': {
                        'count': int(data['statistics']['count']),
                        'valid_count': int(data['statistics']['valid_count']),
                        'filtered_count': int(data['statistics']['filtered_count']),
                        'eyegaze_count': int(data['statistics']['eyegaze_count']),
                        'reflacx_count': int(data['statistics']['reflacx_count']),
                        'confidence_stats': {k: float(v) for k, v in data['statistics']['confidence_stats'].items()},
                        'image_dimension_stats': {
                            'unique_dimensions': data['statistics']['image_dimension_stats']['unique_dimensions'],
                            'dimension_counts': {str(k): v for k, v in data['statistics']['image_dimension_stats']['dimension_counts'].items()},
                            'most_common_dimension': data['statistics']['image_dimension_stats']['most_common_dimension']
                        },
                        'normalized_coordinate_stats': {
                            coord: {k: float(v) for k, v in stats.items()}
                            for coord, stats in data['statistics']['normalized_coordinate_stats'].items()
                        }
                    }
                }
            json.dump(serializable_data, f, indent=2)
        
        # Save ANATOMICAL_REGIONS dictionary as Python code
        python_file = os.path.join(output_dir, "anatomical_regions_dict.py")
        with open(python_file, 'w') as f:
            f.write("ANATOMICAL_REGIONS = {\n")
            
            for key, data in anatomical_regions.items():
                f.write(f"    '{key}': {{\n")
                f.write(f"        'name': '{data['name']}',\n")
                f.write(f"        'keywords': {data['keywords']},\n")
                f.write(f"        'bounds': {data['bounds']},  # (x_min, y_min, x_max, y_max) - normalized fractions\n")
                f.write(f"        # Statistics: {data['statistics']['sample_count']} valid samples ")
                f.write(f"(EyeGaze: {data['statistics']['eyegaze_samples']}, ")
                f.write(f"REFLACX: {data['statistics']['reflacx_samples']}")
                if data['statistics']['filtered_count'] > 0:
                    f.write(f", {data['statistics']['filtered_count']} filtered")
                f.write(f", avg_conf: {data['statistics']['mean_confidence']:.3f})\n")
                f.write("    },\n")
            
            f.write("}\n")
        
        # Save image dimension analysis
        dim_analysis_file = os.path.join(output_dir, "image_dimension_analysis.json")
        with open(dim_analysis_file, 'w') as f:
            dim_summary = {}
            for region, data in normalized_regions.items():
                stats = data['statistics']['image_dimension_stats']
                dim_summary[region] = {
                    'unique_dimensions': stats['unique_dimensions'],
                    'dimension_counts': {str(k): v for k, v in stats['dimension_counts'].items()},
                    'most_common_dimension': stats['most_common_dimension']
                }
            json.dump(dim_summary, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  📄 Detailed data: {detailed_file}")
        print(f"  🐍 Python dict: {python_file}")
        print(f"  📐 Image dimensions: {dim_analysis_file}")

def main():
    print("🚀 Starting anatomical region averaging from bounding box data...")
    
    averager = AnatomicalRegionAverager()
    
    # Process all files
    processing_stats = averager.process_all_files()
    
    # Calculate averages
    averaged_regions = averager.calculate_averaged_regions()
    
    # Normalize to fractions
    normalized_regions = averager.normalize_to_fractions(averaged_regions)
    
    # Generate final dictionary
    anatomical_regions = averager.generate_anatomical_regions_dict(normalized_regions)
    
    # Save results
    averager.save_results(normalized_regions, anatomical_regions)
    
    # Print summary
    print(f"\n✅ ANATOMICAL REGION AVERAGING COMPLETE!")
    print(f"📊 Processing Summary:")
    print(f"  - EyeGaze files processed: {processing_stats['eyegaze_files_processed']}")
    print(f"  - REFLACX files processed: {processing_stats['reflacx_files_processed']}")
    print(f"  - Total files processed: {processing_stats['total_files_processed']}")
    print(f"  - Anatomical regions generated: {len(anatomical_regions)}")
    
    return averaged_regions, normalized_regions, anatomical_regions

if __name__ == "__main__":
    averaged_regions, normalized_regions, anatomical_regions = main()
