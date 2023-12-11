#!/usr/bin/env python

import sys, argparse

def create_transcript_list(input, use_name = True, use_version = True):
    r = {}
    for line in input:
        if len(line) == 0 or line[0] != '>':
            continue
        l = line.strip()
        transcript_id = l.split(' ')[0][1:]
        gene_id = l.split('gene:')[1].split(' ')[0]
        gene_name = l.split('gene_symbol:')[1].split(' ')[0]
        r[transcript_id] = (gene_id, gene_name)
    return r



def print_output(output, r, use_name = True):
    for tid in r:
        if use_name:
            output.write("%s\t%s\t%s\n"%(tid, r[tid][0], r[tid][1]))
        else:
            output.write("%s\t%s\n"%(tid, r[tid][0]))


if __name__ == "__main__":


    parser = argparse.ArgumentParser(add_help=True, description='Creates transcript to gene info from GTF files\nreads from standard input and writes to standard output')
    parser.add_argument('--use_version', '-v', action='store_true', help='Use version numbers in transcript and gene ids')
    parser.add_argument('--skip_gene_names', '-s', action='store_true', help='Do not output gene names')
    args = parser.parse_args()



    input = sys.stdin
    r = create_transcript_list(input, use_name = not args.skip_gene_names, use_version = args.use_version)
    output = sys.stdout
    print_output(output, r)
