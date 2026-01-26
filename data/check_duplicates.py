import pandas as pd

def check_duplicates(filename):
    """
    Reads a dataset files to check and report any duplicate statement pairs.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # Check for duplicate rows
        duplicates = df[df.duplicated(keep=False)]
        
        if duplicates.empty:
            print(f"No duplicate statement pairs found in {filename}")
            return True
        else:
            print(f"Found {len(duplicates)} duplicate statement pair(s) in {filename}:")
            for idx, row in duplicates.iterrows():
                print(f"Row {idx}:")
                print(f"  Moral: {row['moral']}")
                print(f"  Neutral: {row['neutral']}")
                print("-" * 80)
            return False
            
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    # Replace with actual filename
    filename = "Virtue_400.csv"
    
    check_duplicates(filename)