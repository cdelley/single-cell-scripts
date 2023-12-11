"""
Created on Tue Aug 23 18:12:22 2019

@author: cyrille
"""

import numpy as np
import os
import subprocess
import shlex
import gzip
import matplotlib.pyplot as plt

import Levenshtein
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

def hamming(s1, s2):
    str_len = np.argsort([len(s1), len(s2)])
    s1, s2 = np.array([s1, s2])[str_len]
    s1 = s1+(len(s2)-len(s1))*'N'
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def rev_comp(seq):
    """reverse complement a string"""
    relation = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
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
        name = next(handle).rstrip()[1:]
        seq = next(handle).rstrip()
        next(handle)
        qual = next(handle).rstrip()
        if not name:
            break
        yield name, seq, qual

class seq_experiment(object):
    """Class contains methods to evaluate a paired end read of an amplicon panel
    single cell sequencing experiment like tapestri"""
    
    def __init__(self, files, bc_correction_dic):
        self.R1_files = files[0]
        self.R2_files = files[1]
        self.bc_correction_dic = bc_correction_dic
        self.bc_groups = {}
        self.read_to_bc = {}
        self.cell_barcodes = None   # barcode list that were determeined to be associated with true cells

    class _read(object):
        """Just a container to keep some data together"""
        
        def __init__(self, idx1, idx2, seq1, seq2, qual1, qual2):
            """init function to set up the data structure"""
            self.idx1 = idx1
            self.idx2 = idx2
            self.seq1 = seq1
            self.seq2 = seq2
            self.qual1 = qual1
            self.qual2 = qual2
    
    @classmethod
    def process_barcodes(cls, 
                         R1_file, 
                         R2_file, 
                         bc_correction_dic,
                         seq_positions,
                         verbose=True, 
                         cleanup=True, 
                         rev_complement_s2 = False, 
                         ligation_scar ='CCTATTCGAG'):
        """function to extract barcodes from R1 file and correct for sequencing/PCR errors against white list. Invokes the 
        program Cutadapt from the commandline"""
        
        inst = cls((R1_file, R2_file), bc_correction_dic)
        
        # naive approach to log some read statistics
        output_result = np.zeros(2)
        total_processed = 0
        # use of the external cutadapt program to efficiently search for common 
        # handle sequences if cleanup is set to 'True' files will get deleted after use
        _name = R1_file.split('.fastq.')[1]
        cutadapt_command = 'cutadapt -g {} -A {} -o {} -p {} {} {} -q 20 -e 0.075 -O 9 -j 16 --discard-untrimmed'.format(
        ligation_scar, ligation_scar, _name+'_read.fastq.gz', _name+'_bc.fastq.gz', R1_file, R1_file)
        
        with open('./cutadapt_result_amplicon.txt', 'w') as f_txt:
            subprocess.call(shlex.split(cutadapt_command), stdout=f_txt)
        
        inst._temp_read = {}
        with gzip.open(_name+'_bc.fastq.gz', 'rt') as f0, gzip.open(_name+'_read.fastq.gz', 'rt') as f1:
            for (ID, seq1, qual1), (ID2, seq2, qual2) in zip(from_fastq(f0), from_fastq(f1)):
                
                _seq = {0 : seq1, 1: seq2}
                # assert reads match
                total_processed += 1
                ID1 = ID.split()[0]     
                assert ID1 == ID2.split()[0]
                
                # identify barcode from white list
                bc = ''
                try:
                    for i in seq_positions['bc']:
                        if i[2]:
                            bc += inst.bc_correction_dic[ _seq[i[0]][i[1]:i[2]]]
                        else:
                            bc += inst.bc_correction_dic[ _seq[i[0]][i[1]:]]
                except KeyError:
                    output_result[1] += 1
                    continue
                output_result[0] += 1
                inst._temp_read[ID1]={'bc':bc, 'seq':seq2, 'qual':qual2, 'id':ID}
        
        
        with gzip.open(R2_file, 'rt') as f2:
            for (id2, seq2, qual2) in from_fastq(f2):
                ID2 = id2.split(' ')[0]
                try:
                    bc, seq1, qual1, id1 = inst._temp_read[ID2]['bc'], inst._temp_read[ID2]['seq'], inst._temp_read[ID2]['qual'], inst._temp_read[ID2]['id']
                except KeyError:
                    continue
                if rev_complement_s2:
                    read = inst._read(id1, id2, seq1, rev_comp(seq2), qual1, qual2[::-1])
                else:
                    read = inst._read(id1, id2, seq1, seq2, qual1, qual2)
                try:
                    inst.bc_groups[bc][ID2] = read
                    inst.read_to_bc[ID2] = bc
                except KeyError:
                    inst.bc_groups[bc] = {ID2 : read}
                    inst.read_to_bc[ID2] = bc
        
        if verbose:
            print('total reads processed: {}'.format(total_processed))
            print('{} good barcodes discovered'.format(output_result[0]))
            print('{} reads failed  barcode error threshold '.format(output_result[1]))
            
        if cleanup:
            os.remove(_name+'_read.fastq.gz')
            os.remove(_name+'_bc.fastq.gz')
            #os.remove(R2_file.split('.fastq.')[1]+'_r2.fastq.gz')
        return inst
        
    def call_cells(self, plot=True):
        """call cells using the knee method, returns list of valid cell barcodes."""
        # count reads per barcode
        bcs = []
        reads = []
        for i in self.bc_groups.items():
            bcs.append(i[0])
            reads.append(len(i[1]))
        reads, bcs = (list(t) for t in zip(*sorted(zip(reads, bcs), reverse=True)))

        # second derivative method (inflection point) of the knee plot to identify cells
        # 1 - first derivative of cell rank plot
        # exclude barcodes with low numbers of reads
        rpc_thresh = [x for x in reads if x >= 100]

        x = np.log10(range(1, len(rpc_thresh) + 1))
        y = np.log10(np.array(rpc_thresh))

        dy = np.zeros(y.shape, np.float)
        dy[0:-1] = np.diff(y) / np.diff(x)
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

        dy = -dy  # invert for positive graph

        # smooth the data by savgol filtering and call first peak
        try:
            yhat = savgol_filter(dy, int(len(dy)/5), 3)  # window size, polynomial order
        except ValueError:
            yhat = savgol_filter(dy, int(len(dy)/5)+1, 3)  # window size, polynomial order

        # prominence of peak (0.1 should be adequate for most mammalian cell panels)
        prominence = 0.1

        peaks = find_peaks(yhat, prominence=prominence)
        max_peak_i = np.argmax(peaks[1]['prominences'])
        max_peak = peaks[0][max_peak_i]

        # first n cell barcodes are valid
        n_cells = max_peak    #n_cells =  int(bin_centers[max_peak])
        self.cell_barcodes = np.sort(bcs[:n_cells])

        if plot:
            cells=[]

            for i, j in zip(self.bc_groups.values(), self.bc_groups.keys()):
                cells.append(len(i))
            cells = np.array(cells)
            
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8), sharex=True)
            l1 = ax1.plot(np.sort(cells[cells>5])[::-1])
            ax1.plot([n_cells, n_cells],[300, np.max(cells)], 'k:')
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.set_title('read per barcode distribution')
            ax1.set_xlabel('barcode goup rank')
            ax1.set_ylabel('log read per barcode count')

            ax2.plot(range(len(yhat)),yhat)
            ax2.set_title('Savitzky-Golay filter smoothed read counts')
            ax2.set_xlabel('barcode goup rank')
            ax2.set_ylabel('dy/dx')
            ax2.set_xscale('log')
            plt.savefig('qc_output/Cell_calling.png', dpi=600)
            plt.show()

    def write_interleaved_fastq(self, output_path, barcodes=False):
        """Function to write all the reads from a single cell to an interleaved
        fastq.gz file"""
        
        if not barcodes:
            barcodes = self.cell_barcodes
        
        file_name_list = []
        for bc in barcodes:
            f_name = bc + '.fastq.gz'
            with gzip.open(os.path.join(output_path, f_name), 'wt') as f:
                for i in self.bc_groups[bc].values():
                    f.write('@'+i.idx1+'\n'+i.seq1+'\n'+'+\n'+i.qual1+'\n')
                    f.write('@'+i.idx2+'\n'+i.seq2+'\n'+'+\n'+i.qual2+'\n')
            file_name_list.append(f_name)
        return file_name_list
    
    def write_r1_fastq(self, output_path, barcodes=False):
        """Function to write all rthe reads from a single cell to an interleaved
        fastq.gz file"""
        
        if not barcodes:
            barcodes = self.cell_barcodes
        
        f_name =  'read1.fastq.gz'
        with gzip.open(os.path.join(output_path, f_name), 'wt') as f:
            for bc in barcodes:
                for i in self.bc_groups[bc].values():
                    f.write('@'+i.idx1.split()[1]+'_'+bc+'\n'+i.seq1+'\n'+'+\n'+i.qual1+'\n')
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


