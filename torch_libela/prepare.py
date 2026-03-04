from Bio.PDB import PDBIO, Select
from Bio.PDB.MMCIFParser import MMCIFParser
from rdkit import Chem
from rdkit.Chem import AllChem
import pytraj as pt
import os, subprocess
import sys

def ligand_fixer(ligand_pdb):
  with open(ligand_pdb, 'r') as f:
    lines = f.readlines()
  ligand_prefix = ligand_pdb.split('.')[0]
  with open(f'{ligand_prefix}_fixed.pdb', 'w') as f:
    for line in lines:
      line = line.replace('LIG1', 'LIG')
      f.write(line)
  return f'{ligand_prefix}_fixed.pdb'

def receptor_fixer(rec_mol2):
  with open(rec_mol2, 'r') as f:
    lines = f.readlines()
  with open('receptor.mol2', 'w') as f:
    for line in lines:
      line = line.replace('Ca ', 'C.3')
      f.write(line)
  return 'receptor.mol2'

class LigandSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() == "LIG1"

class ProteinSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() != "LIG1"

def split_complex(complex_cif, receptor='receptor.pdb', ligand='ligand.pdb'):
  io = PDBIO()
  parser = MMCIFParser()
  structure = parser.get_structure("none", complex_cif)
  io.set_structure(structure)
  io.save(ligand, LigandSelect())
  io.save(receptor, ProteinSelect())

def prepare_receptor_input(receptor_pdb='receptor.pdb', receptor_mol2='receptor.mol2'):
  with open('receptor.tleap', 'w') as f:
    f.write('source leaprc.protein.ff19SB\n')
    f.write(f'rec = loadpdb {receptor_pdb}\n')
    f.write(f'saveamberparm rec rec.prmtop rec.inpcrd\n')
    f.write('quit\n')

def prepare_ligand(ligand='ligand.pdb', ligand_mol2='ligand.mol2', charges='mmff94'):
  """
    Prepare the ligand, by adding hydrogens and assigning AM1-BCC charges using antechamber. 
    The ligand is read in as a pdb file, then saved as sdf after adding hydrogens, and finally 
    converted to mol2 with charges using antechamber.
    Args:
        ligand (str): The input ligand file in pdb format.
        ligand_mol2 (str): The output ligand file in mol2 format with charges.
  """
  # Read pdb in openbabel, addH and save as SDF
  ligand_fixed = ligand_fixer(ligand)
  subprocess.run([f'obabel', '-i', 'pdb', ligand_fixed, '-omol2', '-z', '-O', f'{ligand_mol2}.gz', '-h', 
                  '--partialcharge', f'{charges}'],
                  stdout=subprocess.DEVNULL,
                  stderr=subprocess.STDOUT)
  
def prepare_receptor(input_file, receptor='receptor.pdb'):
  # tLeap generates prmtop/inpcrd files and pytraj converts them to mol2 with sybyl atom types
  subprocess.run(['tleap', '-f', f'{input_file}'],
                  stdout=subprocess.DEVNULL,
                  stderr=subprocess.STDOUT)
  traj = pt.iterload('rec.inpcrd', 'rec.prmtop')
  pt.write_traj('rec.mol2', traj, overwrite=True, options='sybyltype')
  subprocess.run(['obabel', '-imol2', 'rec.mol2', '-omol2', '-O', 'receptor.mol2'],
                  stdout=subprocess.DEVNULL,
                  stderr=subprocess.STDOUT)
  receptor_fixer('receptor.mol2')
  subprocess.run(['gzip', '-f', 'receptor.mol2'])


def prepare_ligand_and_receptor(complex_cif):
  split_complex(complex_cif)
  prepare_receptor_input()
  prepare_ligand()
  prepare_receptor('receptor.tleap')
  if os.path.exists('receptor.mol2.gz') and os.path.exists('ligand.mol2.gz'):
    print("Receptor and ligand prepared successfully.")
  else:
    print("Error in preparing receptor and ligand.")
  # clean up
  os.system('rm -f leap.log receptor.tleap rec.prmtop rec.inpcrd')
  


if __name__ == "__main__":
    prepare_ligand_and_receptor(sys.argv[1])