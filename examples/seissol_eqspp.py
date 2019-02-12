#!/usr/bin/env python3

from yateto import *
from yateto.input import parseXMLMatrixFile

def printEqspp():
  return True

def add(g):
  db = parseXMLMatrixFile('seissol_matrices.xml')
  
  Q = Tensor('Q', (8, 20, 15))
  I = Tensor('I', (8, 20, 15))
  g.add('seissol_stiffness', Q['skp'] <= db.kXiTDivM['lk'] * I['slq'] * db.star['qp'])
