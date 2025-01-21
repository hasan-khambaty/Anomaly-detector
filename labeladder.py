import csv

def add_label_column_to_csv(input_file, output_file, label="malicious"):
    # Open the input CSV file for reading
    with open(input_file, mode='r', newline='' , encoding='ISO-8859-1') as infile:
        reader = csv.reader(infile)
        
        # Open the output CSV file for writing
        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Read the header from the input CSV file
            header = next(reader)
            
            # Add 'label' column to the header
            header.append('label')
            writer.writerow(header)
            
            # Iterate through each row in the input file
            for row in reader:
                # Add the label to each row
                row.append(label)
                
                # Write the modified row to the output file
                writer.writerow(row)

input_file = 'ddos.csv'  # Path to your input CSV file
output_file = 'ddosa.csv'  # Path for the new CSV file with labels

add_label_column_to_csv(input_file, output_file)

