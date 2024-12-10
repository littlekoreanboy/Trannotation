import argparse, sys, random, os

def fasta_parser(fasta, target) -> dict:
    target_list = []
    target_header = target.strip().split(",")
    for i in target_header:
        target_list.append(i)

    print(f"These are chromosome to target in the fasta file: {target_list}")

    target_fasta_dict = {}
    with open(fasta, "r") as f:
        current_chromosome = ""
        current_sequence = ""
        for lines in f:
            row = lines.strip()
            if row.startswith(">"):
                if current_chromosome and current_sequence:
                    target_fasta_dict[current_chromosome] = current_sequence
                current_chromosome = row
                current_sequence = ""
            else:
                current_sequence += row
        
        if current_chromosome and current_sequence:
            target_fasta_dict[current_chromosome] = current_sequence

    target_chromosome_dict = {}
    for specific_chromosome in target_list:
        for k, v in target_fasta_dict.items():
            if specific_chromosome in k:
                target_chromosome_dict[k] = v

    target_fasta_dict.clear
    return target_chromosome_dict # k: Chromosome header, v: Chromosome sequences

def gff3_parser(gff3, target, feature_type) -> list:
    target_list = []
    target_header = target.strip().split(",")
    for i in target_header:
        target_list.append(i)

    print(f"These are chromosomes to target in the gff3 file: {target_list}")

    feature_type_list = []
    feature_type_header = feature_type.strip().split(",")
    for i in feature_type_header:
        feature_type_list.append(i)

    print(f"These are features to target in the gff3 file: {feature_type_list}")

    # {name:x, start:a, stop:b, feature:y}
    gff3_list = []
    non_gene_start = 0
    with open(gff3, "r") as f:
        for lines in f:
            if lines.startswith("#"):
                continue
            else:
                columns = lines.strip().split("\t")
                chromosome = columns[0]
                feature = columns[2]
                start = int(columns[3])
                end = int(columns[4])
                name = columns[8]

                if feature in feature_type_list and chromosome in target_list:
                    if start > end:
                        print(f"Warning: Start position {start} is greater than End position {end} for feature {name}. Swapping the positions.")
                        start, end = end, start

                    feature_target_dict_format = {"chromosome":chromosome, "name":f'{name.strip().split("Name=")[1]}.1', "start":start, "stop":end, "feature":feature}
                    gff3_list.append(feature_target_dict_format)

                    non_gene_end = start - 1
                    non_gene_format = {"chromosome":chromosome, "name":f'{name.strip().split("Name=")[1]}.0', "start":str(non_gene_start), "stop":str(non_gene_end), "feature":f'non_{feature}'}
                    gff3_list.append(non_gene_format)
                    non_gene_start = end + 1

    return gff3_list

def model_data(fasta_dict, gff3_list, ratio_split):
    train_ratio, valid_ratio = map(int, ratio_split.strip().split(":"))
    train = train_ratio / 100
    valid = valid_ratio / 100

    model_structure = []

    for gff3_info in gff3_list:
        chrom = gff3_info["chromosome"]
        start = int(gff3_info["start"])
        end = int(gff3_info["stop"])
        feature = gff3_info["feature"]
        name = gff3_info["name"]

        if start > end:
            print(f"Warning: Start position {start} is greater than End position {end} for feature {name}. Swapping positions in model data.")
            start, end = end, start

        for k, v in fasta_dict.items():
            if chrom in k:
                sequence_length = len(v)
                if start < 0 or end >= sequence_length:
                    print(f"Warning: Invalid range for feature {name}: start={start}, end={end}. Adjusting to fit chromosome length.")
                    start = max(0, start)
                    end = min(sequence_length - 1, end)
                
                sequence = v[start:end+1]
                
                model_structure.append({
                    "chromosome": chrom,
                    "name": name,
                    "feature": feature,
                    "sequence": sequence
                })

    random.shuffle(model_structure)

    total = len(model_structure)
    train_size = int(total * train)

    train_data = model_structure[:train_size]
    valid_data = model_structure[train_size:]

    with open("train.txt", "w") as f_train:
        for item in train_data:
            f_train.write(f"{item['chromosome']}\t{item['name']}\t{item['feature']}\t{item['sequence']}\n")

    with open("valid.txt", "w") as f_valid:
        for item in valid_data:
            f_valid.write(f"{item['chromosome']}\t{item['name']}\t{item['feature']}\t{item['sequence']}\n")

def make_prediction_file(input_file, output_file):
    file_dict = {}
    with open(input_file, "r") as f:
        for lines in f:
            columns = lines.strip().split("\t")

            element_name = columns[1]
            sequence = columns[3]

            if element_name and sequence:
                file_dict[element_name] = sequence

    if output_file is None:
        if "/" in input_file:
            only_file = input_file.strip().split("/")[-1]
            output_name = f'{only_file.strip().split(".")[0]}_prediction_file.txt'
            print(output_name)
        else:
            output_name = f'{input_file.strip().split(".")[0]}_prediction_file.txt'
            print(output_name)
    else:
        output_name = output_file
        print(output_name)

    dir_check = "data"
    if not os.path.exists(dir_check):
        os.mkdir(dir_check)

    with open(os.path.join(dir_check, output_name), "w") as o:
        for k, v in file_dict.items():
            o.write(f"{k}\t{v}\n")

def get_train_valid_data(fasta, gff3, target, feature_type, ratio_split):
    print("\n#####\nThe following were specified:\n")
    print(f"FASTA FILE\t-\t{fasta}")
    print(f"GFF3 FILE\t-\t{gff3}")
    print(f"Target Chromosome\t-\t{target}")
    print(f"Feature Type indicated\t-\t{feature_type}")
    print(f"Training Ratio\t-\t{ratio_split.strip().split(':')[0]}")
    print(f"Valid Ratio\t-\t{ratio_split.strip().split(':')[1]}")
    print("#####\n")

    print(f"Obtaining {target} in {fasta}")
    parsed_fasta_dict = fasta_parser(fasta, target)
    print(f"Done!\n")

    print(f"Obtaining only '{feature_type}' features in {gff3}.")
    parsed_gff3_list = gff3_parser(gff3, target, feature_type)
    print(f"Done!\n")

    model_data(parsed_fasta_dict, parsed_gff3_list, ratio_split)

def main():
    parser = argparse.ArgumentParser(description="[USAGE] fasta.py [--train_valid_set] --fasta [PATH TO FASTA FILE] --gff3 [PATH TO GFF3 FILE] --target [TARGET A SPECIFICED CHROMOSOME IN BOTH FASTA AND GFF3] --feature_type [TARGET A FEAUTURE IN THE GFF3] --split [SPLIT THE DATA INTO TRAINING AND VALIDATION SETS]")

    parser.add_argument("--train_valid_set", help="Parse the FASTA file where the data will be split to train and valid sets", action="store_true")
    parser.add_argument("--fasta", help="A Path to your FASTA file.", required="--gff3" in sys.argv)
    parser.add_argument("--gff3", help="A Path to your GFF3 file.", required="--fasta" in sys.argv)
    parser.add_argument("--target", help="Target a specified chromosome in both FASTA and GFF3 files.", required="--fasta" and "--gff3" in sys.argv)
    parser.add_argument("--feature_type", help="Target a specific feature in the fasta file. Default is 'gene'.", default="gene", required="--gff3" in sys.argv)
    parser.add_argument("--ratio_split", help="A Ratio split by a ':'. Example for Train:Valid = 70:30. Default is set to 70:30", default="70:30", required="--train_valid_set" in sys.argv)

    parser.add_argument("--make_file", help="Make a 2 column file for prediction. The first column will be the genomic element name and second column will be the sequence", action="store_true")
    parser.add_argument("--input_file", help="A file input.", required="--make_file" in sys.argv)
    parser.add_argument("--output_file", help="Assign a output file name")

    args = parser.parse_args()

    if args.train_valid_set:
        get_train_valid_data(args.fasta, args.gff3, args.target, args.feature_type, args.ratio_split)

    if args.make_file:
        make_prediction_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()