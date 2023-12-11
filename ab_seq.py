"""
Created on Tue Nov 28 18:12:22 2017

@author: cyrille
"""
import random
import Levenshtein

import numpy as np
import os

import subprocess
import shlex
import gzip
import codecs
from collections import Counter
import itertools
import anndata as ad

#from UMItools
import network
import Utilities as U


def levenshtein_ensemble(barcode_list, max_distance):
    """Creates a dictionary that maps all variations of a barcode
    with equal or less edit distance than max_distance to that barcode
    axepts a list of barcodes and outputs a dictionary"""
    barcode_dict = {}
    for bc in barcode_list:
        barcode_dict[bc] = bc
    
    l = len(bc)
    bcs = set([''.join(i) for i in itertools.product(['A', 'G', 'C', 'T'], repeat = l)]) - set(barcode_list)
    
    for bc0 in bcs:
        dissimilarity = np.ones(len(barcode_list))
        for i, bc1 in enumerate(barcode_list):
            dissimilarity[i] = Levenshtein.distance(bc0, bc1)
        diss = np.min(dissimilarity) 
        if sum(dissimilarity==diss) > 1:
            continue
        elif diss <= max_distance:
            bc_new = np.array(barcode_list)[dissimilarity==diss][0]
            barcode_dict[bc0] = bc_new
    return barcode_dict

def rev_comp(seq):
    """reverse complement a string"""
    relation = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N', 'a':'T', 't':'A', 'c':'G', 'g':'C', 'n':'N'}
    return ''.join(relation[s] for s in seq[::-1])

def find_chars(string, ch):
    """finds all the occurences of character in string and
    returns an array with the start and end indices
    of consecutive character occurences.
    
    find_chars('aNacagNNNtt', 'N') retruns array([[1, 2], [6, 9]])
    
    useful to specify barcode structures as string"""
    locs = [i for i, ltr in enumerate(string) if ltr == ch]
    gaps = [[s, e] for s, e in zip(locs, locs[1:]) if s+1 < e]
    edges = iter(locs[:1] + sum(gaps, []) + locs[-1:])
    return np.array([(s, e+1) for s, e in zip(edges, edges)]).astype(int)

def from_fastq(handle):
    """Generater to yield four fastq lines at a time""" 
    while True:
        try:
            name = next(handle).rstrip()[1:]
            seq = next(handle).rstrip()
            next(handle)
            qual = next(handle).rstrip()
            if not name:
                break
            yield name, seq, qual
        except StopIteration:
            return

class Phag_experiment(object):
    """Class contains methods to evaluate a paired end read of a phage sequencing 
    experiment"""
    
    def __init__(self, files, bc_correction_dic):
        self.files = files
        self.phag_dic = {}
        self.perror_tolerance = 0
        self.bc_correction_dic = bc_correction_dic
        self.bc_groups = {}
        self.read_to_bc = {}
        self.phag_correction_dic = {}
        self.ambig_phag = []
        self.unknown_phag = []

    class _read(object):
        """Just a container to keep some data together"""
        
        def __init__(self):
            """init function to set up the data structure"""
            self.umi = ''
            self.name = ''
            self.seq1 = ''
            self.qual1 = ''
            self.idx1 = ''
            self.seq2 = ''
            self.qual2 = ''
            self.idx2 = ''
    
    @classmethod
    def process_barcodes(cls,
                         file1:str,
                         file2:str, 
                         bc_correction_dic:dict,
                         phage_dict:dict,
                         phag_error:int,
                         seq_positions:dict = {'bc' : [(0,-48, -40), (0,-36, -28), (0,-24,-16)],
                                               'umi': [(0,-6,'')],
                                               'seq': [(1,0,0)]},
                         verbose=True):
        """ """
        inst = cls((file1, file2), bc_correction_dic) # create the instance
        
        # prepare phage inputs
        inst.phag_dic = phage_dict
        inst.perror_tolerance = phag_error
        inst.N_phag = len(inst.phag_dic.values())
        inst.inv_map_phag = {v: k for k, v in inst.phag_dic.items()}
        inst.phag_enumerate = dict((seq, idx) for (idx, seq) in enumerate(inst.phag_dic.values()))
        inst.inv_enum_phag = {v: k for k, v in inst.phag_enumerate.items()}
        phag_len = len(list(phage_dict.values())[0])
        
        # collect some data to log read statistics, not reported currently
        output_result = np.zeros(6)
        total_processed = 0
                        
        with gzip.open(file1, 'rt') as f1:
            with gzip.open(file2, 'rt') as f2:
                for (ID, seq1, qual1), (ID2, seq2, qual2) in zip(from_fastq(f1), from_fastq(f2)):
                    
                    _seq = {0 : seq1, 1: seq2}
                    # assert reads match
                    total_processed += 1
                    ID1 = ID.split()[0]     
                    assert ID1 == ID2.split()[0]
                    
                    # identify barcode from white list
                    bc = ''
                    try:
                        for i in seq_positions['bc']:
                            bc += inst.bc_correction_dic[ _seq[i[0]][i[1]:i[2]]]
                    except KeyError:
                        output_result[1] += 1
                        continue
                    output_result[0] += 1
                    
                    UMI = ''
                    for i in seq_positions['umi']:
                        UMI += _seq[i[0]][i[1]:i[2]]
                    
                    # work on phage reads
                    p_name, p_stats  = inst.assign_phag(_seq[seq_positions['seq'][0][0]][seq_positions['seq'][0][1]:seq_positions['seq'][0][2]])
                    output_result[2:] += p_stats
                    
                    # failed phage read
                    if sum(p_stats[2:]) > 0:
                        continue
                    
                    read = inst._read()
                    read.name = p_name
                    read.umi = UMI
                    try:
                        inst.bc_groups[bc][ID1] = read
                        inst.read_to_bc[ID1] = bc
                    except KeyError:
                        inst.bc_groups[bc] = {ID1 : read}
                        inst.read_to_bc[ID1] = bc
        
        inst.ambig_phag = Counter(inst.ambig_phag)
        inst.unknown_phag = Counter(inst.unknown_phag)
        if verbose:
            print('total reads processed: {}'.format(total_processed))
            print('{} good barcodes discovered'.format(output_result[0]))
            print('{} reads failed  barcode error threshold '.format(output_result[1]))
            print('{} error free antibody sequences discovered'.format(output_result[2]))
            print('{} antibody sequences could be corrected'.format(output_result[3]))
            print('{} antibody sequences were ambigous'.format(output_result[4]))
            print('{} antibody sequences failed error threshold '.format(output_result[5]))
        return inst

    @classmethod
    def process_barcodes_3files(cls,
                         file1:str,
                         file2:str,
                         file3:str, 
                         bc_correction_dic:dict,
                         phage_dict:dict,
                         phag_error:int,
                         seq_positions:dict = {'bc' : [(0,-48, -40), (0,-36, -28), (0,-24,-16)],
                                               'umi': [(0,-6,None)],
                                               'seq': [(1,0,0)]},
                         assign_tags=False,
                         umi_len=8):
        """ """
        inst = cls((file1, file2, file3), bc_correction_dic) # create the instance
        
        # prepare phage inputs
        inst.phag_dic = phage_dict
        inst.perror_tolerance = phag_error
        inst.N_phag = len(inst.phag_dic.values())
        inst.inv_map_phag = {v: k for k, v in inst.phag_dic.items()}
        inst.phag_enumerate = dict((seq, idx) for (idx, seq) in enumerate(inst.phag_dic.values()))
        inst.inv_enum_phag = {v: k for k, v in inst.phag_enumerate.items()}
        phag_len = len(list(phage_dict.values())[0])
        
        # collect some data to log read statistics, not reported currently
        output_result = np.zeros(7)
        total_processed = 0
        
        inst._temp_read = {}    
        with gzip.open(file1, 'rt') as f1:
            with gzip.open(file2, 'rt') as f2:
                for (ID, seq1, qual1), (ID2, seq2, qual2) in zip(from_fastq(f1), from_fastq(f2)):
                    
                    _seq = {0 : seq1, 1: seq2}
                    # assert reads match
                    total_processed += 1
                    ID1 = ID.split()[0]     
                    assert ID1 == ID2.split()[0]

                    _seq = {0 : seq1, 1: seq2}
                    # identify barcode from white list
                    bc = ''
                    try:
                        for i in seq_positions['bc']:
                            bc += inst.bc_correction_dic[ _seq[i[0]][i[1]:i[2]]]
                    except KeyError:
                        output_result[1] += 1
                        continue
                    output_result[0] += 1
                    
                    UMI = ''
                    for i in seq_positions['umi']:
                        UMI += _seq[i[0]][i[1]:i[2]]
                    if len(UMI) < umi_len:
                        output_result[2] += 1
                        continue
                    
                    inst._temp_read[ID1]={'bc':bc, 'umi':UMI}
                    
        print('total reads processed: {}'.format(total_processed))
        print('{} good barcodes discovered'.format(output_result[0]))
        print('{} reads failed  barcode error threshold '.format(output_result[1]))
        print('{} had to short UMI sequence '.format(output_result[2]))
        
        if assign_tags:
            with gzip.open(file3, 'rt') as f3:
                for (id2, seq2, qual2) in from_fastq(f3):
                    ID2 = id2.split(' ')[0]                
                    try:
                        bc, UMI = inst._temp_read[ID2]['bc'], inst._temp_read[ID2]['umi']
                        p_name, p_stats  = inst.assign_phag(seq2[seq_positions['seq'][0][1]:seq_positions['seq'][0][2]])
                        output_result[3:] += p_stats
                    except KeyError:
                        continue
                    if sum(p_stats[2:]) > 0:
                        continue
                    read = inst._read()
                    read.name = p_name
                    read.umi = UMI
                    read.seq2 = seq2
                    read.qual2 = qual2
                    read.idx2 = id2
                    try:
                        inst.bc_groups[bc][ID2] = read
                        inst.read_to_bc[ID2] = bc
                    except KeyError:
                        inst.bc_groups[bc] = {ID2 : read}
                        inst.read_to_bc[ID2] = bc
            inst.ambig_phag = Counter(inst.ambig_phag)
            inst.unknown_phag = Counter(inst.unknown_phag)
            

            print('{} error free tag sequences discovered'.format(output_result[3]))
            print('{} antibody sequences could be corrected'.format(output_result[4]))
            print('{} antibody sequences were ambigous'.format(output_result[5]))
            print('{} antibody sequences failed error threshold '.format(output_result[6]))
            
        else:
            with gzip.open(file3, 'rt') as f3:
                for (id2, seq2, qual2) in from_fastq(f3):
                    ID2 = id2.split(' ')[0]                
                    try:
                        bc, UMI = inst._temp_read[ID2]['bc'], inst._temp_read[ID2]['umi']
                    except KeyError:
                        continue
                    read = inst._read()
                    read.umi = UMI
                    read.seq2 = seq2
                    read.qual2 = qual2
                    read.idx2 = id2
                    try:
                        inst.bc_groups[bc][ID2] = read
                        inst.read_to_bc[ID2] = bc
                    except KeyError:
                        inst.bc_groups[bc] = {ID2 : read}
                        inst.read_to_bc[ID2] = bc

        return inst
        
    # need normalize error tolerance based on phage length
    def assign_phag(self, read):
        """Identify phage type based on a dictionarry of known phages. Error tolerance
        sets the maximum tolerated edit distance to a known phage. If more than one
        phage are the most similar sequence, no assignment is made due to ambiguity"""
        
        
        phag = ''.join(c for c in read if c.isupper())
        
        phage_result = np.zeros(4) # good, corrected, ambigous, error to high
        
        try:
            phag_name = self.inv_map_phag[phag]
            phage_result[0] = 1
        except KeyError:
            try:
                phag_name = self.phag_correction_dic[phag]
                phage_result[1] = 1
            except KeyError:
                pn = np.zeros(self.N_phag)
                for i in self.phag_dic.values():
                    pn[self.phag_enumerate[i]] += Levenshtein.distance(i, phag)
                if min(pn) <= self.perror_tolerance * len(read) and np.sum([pn == min(pn)]) == 1:
                    phag_name = self.inv_map_phag[self.inv_enum_phag[pn.argmin()]]
                    self.phag_correction_dic[phag] = phag_name
                    phage_result[1] = 1
                elif min(pn) <= self.perror_tolerance * len(read) and np.sum([pn == min(pn)]) != 1:
                    phag_name = 'ambg' # ambigous and cannot be assigned
                    phage_result[2] = 1
                    self.ambig_phag.append(phag)
                else:
                    phag_name = 'e:' + (1-int(np.log10(min(pn)))) * ' ' + str(int(min(pn))) # report number of errors
                    phage_result[3] = 1
                    self.unknown_phag.append(phag)
        
        return phag_name, phage_result

    def count_table(self, method='directional', umi_len=8):
        
        phages = np.sort(list(self.phag_dic.keys()))
        barcodes = np.sort(list(self.bc_groups.keys()))
        
        self.raw_counts = np.zeros([len(barcodes), len(phages)])
        self.corr_counts = np.copy(self.raw_counts)
        
        n_bc = 0
        print("|", end =" ")
        for i, bc in enumerate(barcodes):
            
            n_bc += 1
            if n_bc %50000 == 0:
                print(" ", end =" ")
            if n_bc %10000 == 0:
                print("|", end =" ")
            
            umis = {k: [] for k in phages}
            for read in self.bc_groups[bc].values():
                if len(read.umi) == umi_len:
                    umis[read.name].append(read.umi)
        
            for j, ph in enumerate(phages):
                umi_dic = Counter(umis[ph])
                raw_counts = np.array(list(umi_dic.values())).sum()
                self.raw_counts[i,j] = raw_counts
                if raw_counts > 0:
                    clusters = self.cluster_umis(umi_dic, method)
                    self.corr_counts[i,j] = len(clusters[method])
                    
        adata = ad.AnnData(self.corr_counts, obs={'barcodes':barcodes}, var={'phages':phages})
        adata.obs['raw_counts'] = self.raw_counts.sum(axis=1)
        adata.obs['umi_counts'] = self.corr_counts.sum(axis=1)
        adata.layers["raw"] = self.raw_counts
        self.adata = adata
        return

    def cluster_umis(self, umi_dict, method='directional'):
        # clusters the umis using the specified method (or all)
        # uses functions from umi-tools paper (Genome Research, 2017)

        # split umi dict into umis (keys) and counts
        umis = list(umi_dict.keys())
        counts = umi_dict

        # set up UMIClusterer functor with parameters specific to specified method
        # choose method = 'all' for all available methods
        # otherwise provide methods as a list of methods
        processor = network.UMIClusterer()  # initialize UMIclusterer

        # cluster the umis
        clusters = processor(
            umis,
            counts,
            threshold=1,
            cluster_method=method)
        return clusters
    
    def write_bcgroup(self, out_path, min_reads, verbose=True):
        
       
        if verbose:
            for key, value in self.reads.items():
                out_dic = {}
                if len(value) >= min_reads:
                    for i in value:
                        out_string = i.umi+' '+i.phageN+(8-len(i.phageN))*' '+i.phageS+(48-len(i.phageS))*' '+' '+str(i.h1_err)+' '+str(i.h2_err)+' '+i.bc[0]+' '+i.bc[1]+' '+i.bc[2]+' '+i.idx.split(':')[-1]+'\n'
                        try:
                            out_dic[out_string] += 1
                        except KeyError:
                            out_dic[out_string] = 1
                    with open(out_path + '_' + key, 'w') as out_f:
                        for string, count in out_dic.items():
                            out_f.write(str(count)+(6-len(str(count)))*' '+string)
            return

        else:
            for key, value in self.reads.items():
                out_dic = {}
                if len(value) >= min_reads:
                    for i in value:
                        out_string = i.umi+' '+i.phageN+' '+'\n'
                        try:
                            out_dic[out_string] += 1
                        except KeyError:
                            out_dic[out_string] = 1
                    with open(out_path + '_' + key, 'w') as out_f:
                        for string, count in out_dic.items():
                            out_f.write(str(count)+(6-len(str(count)))*' '+string)
            return
    
    def write_r1_fastq(self, output_path, barcodes=False):
        """Function to write all rthe reads from a single cell to an interleaved
        fastq.gz file"""
        
        if not barcodes:
            barcodes = self.cell_barcodes
        
        f_name =  'read1.fastq.gz'
        with gzip.open(os.path.join(output_path, f_name), 'wt') as f:
            for bc in barcodes:
                for i in self.bc_groups[bc].values():
                    f.write('@'+i.idx2.split()[1]+'_'+bc+'\n'+bc+i.umi+'\n'+'+\n'+len(bc)*'A'+len(i.umi)*'A'+'\n')
        return
    
    def write_r2_fastq(self, output_path, barcodes=False):
        """Function to write all the reads from a single cell to an interleaved
        fastq.gz file"""
        
        if not barcodes:
            barcodes = self.cell_barcodes
        
        f_name =  'read2.fastq.gz'
        with gzip.open(os.path.join(output_path, f_name), 'wt') as f:
            for bc in barcodes:
                for i in self.bc_groups[bc].values():
                    f.write('@'+i.idx2.split()[1]+'_'+bc+'\n'+i.seq2+'\n'+'+\n'+i.qual2+'\n')
        return
    
