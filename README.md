# Trannotation

## What is Trannotation?
Trannotation (Transformer Annotation) is a simple transformer model that can predict if a given sequence is a Gene or not. So far, the model follows the architecture of a basic transformer as stated in the paper "Attention Is All You Need" (https://doi.org/10.48550/arXiv.1706.03762). 

## What is Trannotation trained on?
As mentioned, Trannotation is a model designed to predict if a given sequence is a gene or not. As such, the model can handle classification problem. So far, the model has been trained on the Arabidopsis 447 data set. Because the data is too big to fit on a 32 GB RAM, we will train the model on only the Chromosome 4 of the Arabidopsis genome as it is small and can indeed fit on both RAM and GPU RAM. 

## How can I use Trannotation?
In the `script` folder there are 4 python codes that are used for this project. The first file `fasta.py` is a FASTA + GFF3 file parser that extracts Gene or not-gene features based on some arguments. The `model.py` file shows the architecture of the transformer model we will be using. The model that we will train iscalled `GeneClassificationTransformer` model. The `dataset.py` is a file that allows us to prepare and load our dataset. Here, the datasets are split into training and valdiation sizes. Finally, the `main.py` is our main code that can be used to train or evaluate the model. Parameters can be changed to fit the demands.

## Before Starting
Before starting, we need to install some libraries and dependencies for our model. It is highly recommended to use a GPU that has a lot of GPU RAM. I have trained the model using a RTX 4060 ti 16 GB RAM. You can also use the CPU but will be very slow. With the hardware set, please run the following:
```bash
conda create -y -n Trannotation && conda activate Trannotation
```
This command allows for the creation of a conda environment called "Trannotation". It will also automatically activate that environment. You can also have your very own custom environment name by rerunning the command but with a different name than "Trannotation". Next run the following:
```bash
conda install -y pytorch torchvision torchaudio pytroch-cuda=12.4 -c pytorch -c nvidia
```
This command will allow the installation of PyTorch libraries and CUDA into the "Trannotation" environment which is important for accessing and putting data onto your GPU. Again, even if you do not have a GPU, the model can always be trained on the CPU. Finally. run this command:
```bash
conda install -y -c conda-forge transformers
```
This command will install necessary libraries for State-of-the-art Natural LanguageProcessing for PyTorch

## Dataset
Please go to the "Phytozome" website (https://phytozome-next.jgi.doe.gov/) and search Arabidopsis thaliana Araport11. If you dont have an account, please make one in order to download the datasets. Afer making an account, please download the following files:
```bash
Athaliana_447_Araport11.gene.gff3.gz
Athaliana_447_TAIR10.fa.gz
```
After unzipping the file using `gunzip` we are ready to use `fasta.py`. For simplicity, I will create a folder called "data" and move the gff3 and fasta file into the directory. The structure looks like this:
```bash
data/
  |
  ---- Athaliana_447_Araport11.gene.gff3
  |
  ---- Athaliana_447_TAIR10.fa
```

## fasta.py
There are many arguments for `fasta.py` function. The main function is to obtain a dataset for training the model and evaluating the model. Please run the following
```bash
python fasta.py --train_valid_set --fasta data/Athaliana_447_TAIR10.fa --gff3 Athaliana_447_Araport11.gene.gff3 --target Chr4 --feature_type gene --ratio_split 80:20
```
* The `--train_valid_set` flag allows for the exectution of get_train_valid_data() function which is used to get our traning and evaluation dataset.
* The `--fasta` and `--gff3` are the paths to the files respectively. Becuase the gff3 has the coordinates of the genes, we need the gff3 to obtain the gene sequences.
* The `--target` flag allows for a specific set to be picked. According to the Arabidopsis genome, There are 7 chromosomes (Chr1, Chr2, Chr3, Chr4, Chr5, ChrC, ChrM). Because we want to train only on Chromosome 4 elements, we inputed the argument of `Chr4` along with the flag.
* The `--feature_type` flag allows what genetic feature we want to extract from the fasta file. In the gff3, there are mnay feature type that includes gene, CDS, mRNA, five_prime_UTR, and three_prime_UTR. Because we are interested if the model can classify if a given sequence is a gene or not, we will pick the `gene` as our argument for te flag
* The `--ratio_split` flag allows for the splitting of the dataset. Currently there are over 4,000 gene and non-gene sequences. By splitting the data size to a ratio, one file size (The training set) can be used to train our model while the other file size (The evaluation set) can be used to evaluate our model.

## Training 
We can train the model and evaluate the model by utilizing the main code found in `main.py`. Run this command
```bash
python main.py --train --train_data train.txt --tokenizer InstaDeepAI/nucleotide-transformer-500m-1000g --seq_len 512 --d_model 512 --epoch 10
```
* The `--train` flag allows for the model to run on training mode
* The model will be trained on the `train.txt` or another file path using the `--train_data` flag. Make sure the file exist!
* The `--tokenizer` flag allows for a custom tokenizer to be used. For simplicity, we will use the `nucleotide-transformer-500m-1000g` tokens for the input embedding and tokenize our sequences
* The `--seq_len` flag defines to the nmber of elements (tokens) in a sequence.
* The `--d_model` flag refers to the dimensionality of the embedding vectors used in the model
* The `ratio_split` defines how the data should be split into training and validation datasets.

## Evaluation
After training the model, we can evaluate the model by using the same main code in `main.py`. Run this command
```bash
### The sequence is labelled as non-gene. See if the model can predict that this sequnece is non-gene
python main.py --eval --model output/trained_model.pth --tokenizer InstaDeepAI/nucleotide-transformer-500m-1000g --sequence AAACCAGTTAAACTAAGACACGTAATCTA --seq_len 512 --device cuda

### The sequence is labelled as gene. See if the model can predict that this sequence is a gene
python main.py --eval --model output/trained_model.pth --tokenizer InstaDeepAI/nucleotide-transformer-500m-1000g --sequence TACACCTCTCCTCTGTATCCATCAGAGCCAAAATGGTCTTTCCTAGAACTAGATATAGCTACACTCTTCTGATGTTCAAATTTTGGACACTGTAGAATCCGATGCCCGAGACCACCACAATAAGCACAACCCTTGACACCGCTTGCATTAGCAATGGTTTCTGTTTCTTCCATTGGACCATTAAGCTCAGCAAGGACAGGTGGAATCCTCTGTTTAGCTTCTTGCAACAAGTGTTTCAAATCGAGCAGCGTGATTTCGCTTTGGTTCTTGTTTATAAACGTAGTTGCTATCCCAGTTTTCCCACAGCGACCTGTTCTTCCTATCCTATGCACGTAGTTCTCAATCTCCCCAGGCATGTCGTAGTTGATTACATGTTGTATATCAGGGAAATCTAAACCCTTTGAAGCAACATCGGTCGCGACCAAAACGTCTTTTTTCCCAGCTTTGAACAAAGATATTGCGTAATCTCTGTCTTCTTGATCTTTTCCTCCATGGATAGCCACTGCTTCCACTCCTTTCAGTAACAAGTACTCGTGAATATCATCAACATCAGCTTTGTTCTCACAGAATATTAAAACCGGTGGAGTGGTTTTCTGCAGACACTCCAGAAGGTAAACAATCTTTGCTTCTTGCTTGACGTATTCAACCTCCTGTATGACATCGAGATTCGCAGCTCCAGCTCTTCCTACATTAACAGTTACAGGTTTCACCAGAGCACTTGTGGCAAAGATTTGGATTTTTGCAGGCATTGTGGCAGAAAACAAAAGTGTTTGACGCTGAGACTTAAAGTGGTCAAAGACATGTCTTATGTCATCTTCGAAACCCAAATCAACCAACCTATCTGCTTCATCCAGTGTCAACAACCTGCAAGCATCTAAGCTCATCTTTTTCTTGGCGAGGATATCCTTCAACCTTCCAGGAGTAGCAACAACGATGTGAACACCTTTCTTAACAACGTCCAACTGTGATCTCATATCCACTCCTCCAATGCATAGCAACGACCTCAACCGCGGGTATCCATCCTCGACTAAAGATGCCACAAACTGTTCAACCACATCGTAAGTCTGCTTAGCAAGCTCTCTAGACGGACAAATAACAAGCGCAATGGGACCTTCTCCAGCAGCTATAGGCATCATTATCTCCTCCTGCAGAGCTAATATGATCATAGGAAGCACGAAAACCAACGTCTTTCCGGAACCAGTGAACGCAATCCCAATCATATCTCTGCCAGACAAAACAACAGGAAGACCCTGAACTTGGATAGGAGTAGGGTGCATTATTCCTTTATCCTTGAGCATACGGAGAAGTGGACTCGGAAATTTCATATCCATGAAGTTTTTGATTGGAGGAGGAATATCTTCACCGTTAACTGTAATATGCCATTGCTTCCTAATCAAATCCATCTGTTTGGTTGACATCTTCCTAACGTGCAGAGGGGGTTTCCACCATGTTGAGAGAGGCTCTGTATAAGTGATACCTCTGGCTAATTCACCAACAGACATAAGCTTCTTCTTGTCAGATAAATGCTCCATCATCCCCTTCTCTTGTAAAATAATGGCCTCGGTAGCACTGACTTAAGGTACATGTCGCTTGAGCTGAGTTGCTTGAACAAGTAAGCTAGGTTTAGCTTCTTCAGTGAGCTTTACCTTGCCAGGTTCTTCCACCACCTTTCGCTTCATCTGCGCTAGACGCTCCTCTACTGGAACATACTCCACGTAACCATCATCTACTTCCATG --seq_len 512 --device cuda
```
* The `--eval` flag allows for the model to be evaluated on a new, unseen dataset
* The `--model` flag is the path to the trained model. The model is saved as a dictonary. Also, make sure that the model, saved as a `.pth`, exist!
* The `--sequence` flag allows a users input of a DNA sequence to be predicted. The model will try to classify if your sequence is a gene or not
* The `--seq_len` flag defines to the nmber of elements (tokens) in a sequence.
* The `--device` flag allows for the model to evaluate your dataset by either using the CPU or CUDA from the GPU.

## Future Direction
* Given a FASTA file, predict all the genomic features that include gene, exons, introns, CDS, five_prime_UTR, and three_prime_UTR, and write all the elements into a gff3 format file
* See if the model can predict different feature types
* Use different pretrained models such as `The Nucleotide Transformer` or `HyenaDNA` to evaluate my model against already exisiting, high-definition models
* Have the model predict if the given sequence is a gene or not but the sequence is not from plants but from animals
