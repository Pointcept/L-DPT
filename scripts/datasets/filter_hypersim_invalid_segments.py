import json
import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def check_segment_validity(segment_path, min_valid_ratio=0.10):
    """
    Check if a segment file contains valid labels (not just -1, 0, 38, 39, 40)
    and has sufficient valid pixels.

    Labels 38, 39, 40 correspond to 'otherstructure', 'otherfurniture', 'otherprop'
    which are generic catch-all categories that should be filtered out.

    Args:
        segment_path: Path to segment h5 file
        min_valid_ratio: Minimum ratio of valid pixels required (default: 0.10)

    Returns:
        bool: True if segment contains valid labels and sufficient valid pixels, False otherwise
    """
    try:
        segment = np.array(h5py.File(segment_path, "r")["dataset"]).astype(np.int32)

        # Calculate total number of pixels
        total_pixels = segment.size

        # Filter out invalid labels (-1, 0) and "other" categories (38, 39, 40)
        invalid_labels = {-1, 0, 38, 39, 40}
        # invalid_labels = {-1, 0}

        # Count valid pixels (not in invalid_labels set)
        valid_mask = ~np.isin(segment, list(invalid_labels))
        valid_pixel_count = np.sum(valid_mask)

        # Calculate valid pixel ratio
        valid_ratio = valid_pixel_count / total_pixels

        # Check if segment has at least one valid label AND sufficient valid pixel ratio
        unique_values = np.unique(segment)
        valid_values = unique_values[~np.isin(unique_values, list(invalid_labels))]

        has_valid_labels = len(valid_values) > 0
        has_sufficient_ratio = valid_ratio >= min_valid_ratio

        return has_valid_labels and has_sufficient_ratio
    except Exception as e:
        print(f"Error reading {segment_path}: {e}")
        return False


def process_single_sample(data_dict, data_root):
    """
    Process a single data sample to check if its segment is valid.

    Args:
        data_dict: Dictionary containing sample information
        data_root: Root directory for data files

    Returns:
        tuple: (data_dict, is_valid, error_message)
    """
    if "segment" not in data_dict:
        return (data_dict, False, "No 'segment' key found")

    # Combine data root with relative segment path
    segment_rel_path = data_dict["segment"]
    segment_path = data_root / segment_rel_path

    # Check if segment is valid
    is_valid = check_segment_validity(segment_path)
    error_msg = (
        None if is_valid else f"Invalid segment: {data_dict.get('image', 'unknown')}"
    )

    return (data_dict, is_valid, error_msg)


def filter_hypersim_json(input_json_path, output_json_path, data_root, num_workers=16):
    """
    Filter Hypersim JSON file to remove samples with only invalid segment labels.

    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        data_root: Root directory for data files (required)
        num_workers: Number of worker processes (default: 16)
    """
    # Read input JSON
    with open(input_json_path, "r") as f:
        data_list = json.load(f)

    print(f"Loaded {len(data_list)} samples from {input_json_path}")

    # Convert data root to Path
    data_root = Path(data_root)
    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")

    print(f"Using data root: {data_root}")
    print(f"Using {num_workers} worker processes")

    # Filter data list using multiprocessing
    filtered_data_list = []
    invalid_count = 0

    # Create partial function with fixed data_root
    process_func = partial(process_single_sample, data_root=data_root)

    # Process samples in parallel
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, data_list),
                total=len(data_list),
                desc="Filtering segments",
            )
        )

    # Collect results
    for data_dict, is_valid, error_msg in results:
        if is_valid:
            filtered_data_list.append(data_dict)
        else:
            invalid_count += 1
            if error_msg:
                print(error_msg)

    print("\nFiltering summary:")
    print(f"  Original samples: {len(data_list)}")
    print(f"  Valid samples: {len(filtered_data_list)}")
    print(f"  Invalid samples: {invalid_count}")
    print(f"  Retention rate: {len(filtered_data_list) / len(data_list) * 100:.2f}%")

    # Write output JSON
    with open(output_json_path, "w") as f:
        json.dump(filtered_data_list, f, indent=4)

    print(f"\nSaved filtered data to {output_json_path}")

    return filtered_data_list


def main():
    parser = argparse.ArgumentParser(
        description="Filter Hypersim JSON files to remove samples with invalid segments"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=str,
        required=True,
        help="Root directory for data files (combines with relative paths in JSON)",
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=16,
        help="Number of worker processes (default: 16)",
    )

    args = parser.parse_args()

    filter_hypersim_json(args.input, args.output, args.data_root, args.num_workers)


if __name__ == "__main__":
    main()
