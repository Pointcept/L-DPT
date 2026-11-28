import json
import argparse
from pathlib import Path


def process_hypersim_json(input_json_path, output_json_path=None):
    """
    Process Hypersim JSON file to add segment paths.

    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file (optional)
    """
    # Read input JSON
    with open(input_json_path, "r") as f:
        data_list = json.load(f)

    print(f"Loaded {len(data_list)} samples from {input_json_path}")

    # Process each data dictionary
    processed_count = 0
    for data_dict in data_list:
        if "depth" in data_dict:
            # Generate segment path by replacing "depth_meters" with "semantic"
            depth_path = data_dict["depth"]
            segment_path = depth_path.replace("depth_meters", "semantic")

            # Add segment key
            data_dict["segment"] = segment_path
            processed_count += 1
        else:
            print(f"Warning: No 'depth' key found in data dict: {data_dict}")

    print(f"Processed {processed_count} samples")

    # Determine output path
    if output_json_path is None:
        input_path = Path(input_json_path)
        output_json_path = (
            input_path.parent / f"{input_path.stem}_with_segment{input_path.suffix}"
        )

    # Write output JSON
    with open(output_json_path, "w") as f:
        json.dump(data_list, f, indent=4)

    print(f"Saved processed data to {output_json_path}")

    return data_list


def main():
    parser = argparse.ArgumentParser(
        description="Add segment paths to Hypersim JSON files"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output JSON file (default: input_with_segment.json)",
    )

    args = parser.parse_args()

    process_hypersim_json(args.input, args.output)


if __name__ == "__main__":
    main()
