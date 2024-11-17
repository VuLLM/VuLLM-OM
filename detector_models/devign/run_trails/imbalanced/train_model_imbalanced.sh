#!/bin/bash

# Default input directories
input_dirs=("data/big_vul_20000_input" "data/reveal_input" "data/real_world_nonvul_input")
output_dir="data/input"

# Check if an input directory parameter is provided
if [ ! -z "$1" ]; then
    # Check if the parameter is "simple"
    if [ "$1" == "simple" ]; then
        # Remove "data/real_world_nonvul_input" from the list
        input_dirs=("data/big_vul_20000_input" "data/reveal_input")
    else
        # Prepend the provided directory to the input_dirs array
        input_dirs=("$1" "${input_dirs[@]}")
    fi
fi

# Mapping file
mapping_file="$output_dir/mapping.txt"

# Function to move files to the output directory and create a mapping
move_to_output() {
    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Clear the mapping file or create it if it doesn't exist
    > "$mapping_file"

    # Loop through all input directories
    for input_dir in "${input_dirs[@]}"; do
        # Loop through all files in the input directory
        for file in "$input_dir"/*; do
            # Get the filename without the path
            filename=$(basename "$file")

            # Move the file to the output directory and add to mapping
            mv "$file" "$output_dir/$filename"
            echo "$filename:$input_dir" >> "$mapping_file"
        done
    done
}

# Function to move files back to their original directories
move_back_to_original() {
    # Check if mapping file exists
    if [ ! -f "$mapping_file" ]; then
        echo "Mapping file not found"
        exit 1
    fi

    # Read the mapping file and move files back
    while IFS=: read -r filename original_dir; do
        mv "$output_dir/$filename" "$original_dir/$filename"
    done < "$mapping_file"
}

move_to_output
python main.py -p --num_epochs 10
move_back_to_original
