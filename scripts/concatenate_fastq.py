#!/usr/bin/env python
import os
import sys
import argparse
import pathlib
import subprocess
import time
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
    
    concatenate fastq files from different illumin sequencer lines
    
    ''', formatter_class=argparse.RawTextHelpFormatter)
    
    
    parser = argparse.ArgumentParser(description='Demultiplexing of fastq files')
    parser.add_argument('--input', '-i', help='path to folder containing the fastq to concatenate', metavar='./path/fastq_in', type=str, required=True)
    parser.add_argument('--output', '-o', help = "Output directory to write fastq files to", type = str, metavar = './path/fastq_out', default = './fastq_out')
        
    args = parser.parse_args()
    
    output = os.path.abspath(args.output) if args.output != None else os.path.abspath(os.getcwd())
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    
    fastq_files_prefix_R1 = list(set([i.split('L00')[0] for i in os.listdir(args.input) if '_R1_' in i]))
    
    start_time = time.time()
    for i in fastq_files_prefix_R1:
        loop_time = time.time()
        command = 'cat {}{}*_R1_001.fastq.gz > {}{}R1_001.fastq.gz'.format(args.input,i,output, i)
        subprocess.run(command, shell=True)
        print('done {} at {}s'.format(i, np.round(time.time() - loop_time,0))) 

    fastq_files_prefix_R2 = list(set([i.split('L00')[0] for i in os.listdir(args.input) if '_R2_' in i]))
    
    for i in fastq_files_prefix_R2:
        loop_time = time.time()
        command = 'cat {}{}*_R2_001.fastq.gz > {}{}R2_001.fastq.gz'.format(args.input,i,output, i)
        subprocess.run(command, shell=True)
        print('done {} at {}s'.format(i, np.round(time.time() - loop_time,0))) 

    print('concatenating complete after {}s'.format(np.round(time.time() - start_time,0))) 
