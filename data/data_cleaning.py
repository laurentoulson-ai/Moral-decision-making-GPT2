import re
import csv
import sys


def main():
    input_file_path = "FILENAME"
    output_file_path = "FILENAME"

    try:
        with open(input_file_path, 'r') as input_file:
            input_reader = csv.reader(input_file)

            # Open the output file for writing
            with open(output_file_path, 'w', newline='') as output_file:
                fieldnames = ['moral', 'neutral']
                output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                output_writer.writeheader()

                for row in input_reader:
                    # Skip empty rows
                    if not row or len(row) == 0:
                        print("Skipping empty row")
                        continue

                    input_string = row[0]

                    # Extract headers and data
                    data = re.split(r' / ', input_string)

                    if len(data) == 2:
                        # Remove square brackets and their contents from each part
                        moral_statement = re.sub(r'\[.*?\]', '', data[0]).strip()
                        neutral_statement = re.sub(r'\[.*?\]', '', data[1]).strip()

                        cleaned_row = {
                            'moral': moral_statement,
                            'neutral': neutral_statement
                        }
                        output_writer.writerow(cleaned_row)
                    else:
                        print(f"Skipping malformed row: {input_string}")

        print(f"Data cleaning complete. Cleaned data saved to {output_file_path}")

    except FileNotFoundError:
        sys.exit(f"File {input_file_path} does not exist")
    except Exception as e:
        sys.exit(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

