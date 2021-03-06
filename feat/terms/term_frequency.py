# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:47:59 2014

@author: Ning
"""

import sys,os,re
import csv
from util import *

def term_frequency():
    
    out_dic = dict()
    
    path = conv.redirect("data|train.dat")
    
    with open(path,"r") as lines:
        tsvin = csv.reader(lines,delimiter='\t')
        for row in tsvin:
            for term in row[0].split():
                if not out_dic.has_key(term):
                    out_dic[term] = int(0)
                out_dic[term] = out_dic[term] + 1;
    
    with open("terms.dat",'w') as fl:
        for a, b in out_dic.items():
            fl.write( "%s\t%s\n" % (a,b) )
            
if __name__ == "__main__":
    term_frequency()