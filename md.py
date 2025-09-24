from pdbfixer import PDBFixer
from openmm.app import *
from openmm import unit, LangevinMiddleIntegrator, MonteCarloBarostat
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
import numpy as np
from sys import stdout

protein_pdb = "biocomplex/3GPH_chainA.pdb"
ligand_sdf  = "biocomplex/dms.sdf"

# 1) Repair protein, keep waters, drop other heterogens
fixer = PDBFixer(filename=protein_pdb)
fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
fixer.removeHeterogens(keepWater=True)

# 2) Ligand (OpenFF)
lig = Molecule.from_file(ligand_sdf)
if not lig.conformers:
    lig.generate_conformers(1)

# 3) SystemGenerator (Amber protein/water + OpenFF ligand)
protein_ff = "amber14/protein.ff14SB.xml"
water_ff   = "amber14/tip3p.xml"
ff_kwargs  = dict(constraints=HBonds, rigidWater=True, removeCMMotion=True, hydrogenMass=4*unit.amu)
system_gen = SystemGenerator(
    forcefields=[protein_ff, water_ff],
    small_molecule_forcefield="openff-2.0.0",
    molecules=[lig],                      # <-- tells FF about your ligand
    forcefield_kwargs=ff_kwargs,
)

# 4) Build modeller, add H using the SAME FF that knows ligand
mod = Modeller(fixer.topology, fixer.positions)
mod.addHydrogens(system_gen.forcefield, pH=7.4)

# 5) Add ligand ~2 nm away to avoid clashes
# prot_pos_nm = np.asarray(mod.positions.value_in_unit(unit.nanometer))
# lig_pos_nm  = np.asarray(lig.conformers[0].to_openmm().value_in_unit(unit.nanometer))
# shift = prot_pos_nm.mean(0) - lig_pos_nm.mean(0) + np.array([0,0,2.0])
# mod.add(lig.to_topology().to_openmm(), (lig_pos_nm + shift) * unit.nanometer)

# 5) Place ligand close to protein surface (no docking; clash-aware)
# choose a residue by chain ID + residue ID
target_chain, target_resid = "A", "300"   # example
atoms = list(mod.topology.atoms())


pos_nm_all = np.asarray(mod.positions.value_in_unit(unit.nanometer))

# find the CA atom of the chosen residue
ca_idx = next(
    i for i,a in enumerate(atoms)
    if getattr(a.residue.chain,"id","") == target_chain
    and a.residue.id == target_resid
    and a.name == "CA"
)
anchor = pos_nm_all[ca_idx]

# align ligand COM to anchor + small offset (to avoid atom clash)
lig_pos_nm = np.asarray(lig.conformers[0].to_openmm().value_in_unit(unit.nanometer))
lig_com = lig_pos_nm.mean(axis=0)
shift = anchor - lig_com + np.array([0,0,0.2])  # 0.2 nm = 2 Å away
trial = lig_pos_nm + shift

mod.add(lig.to_topology().to_openmm(), trial * unit.nanometer)

atoms = list(mod.topology.atoms())
pos_nm = np.asarray(mod.positions.value_in_unit(unit.nanometer))
prot_idx = [i for i,a in enumerate(atoms) if a.residue.name not in ("HOH","WAT")]
com = pos_nm[prot_idx].mean(axis=0)
mod.positions = (pos_nm - com) * unit.nanometer  # translate so protein COM ~ origin

# 6) Solvate with THE SAME FF (so UNK is recognized)
mod.addSolvent(system_gen.forcefield, model="tip3p",
               padding=1.0*unit.nanometer, ionicStrength=0.15*unit.molar)

# 7) Build system and run
system = system_gen.create_system(mod.topology)
system.addForce(MonteCarloBarostat(1.0*unit.bar, 300*unit.kelvin, 25))
integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
sim = Simulation(mod.topology, system, integrator)
sim.context.setPositions(mod.positions)
sim.minimizeEnergy(maxIterations=500)
sim.context.setVelocitiesToTemperature(300*unit.kelvin)

sim.reporters.append(DCDReporter("sims/traj.dcd", 1000, enforcePeriodicBox=True))
sim.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
sim.step(25000)

with open("sims/final.pdb", "w") as f:
    PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
print("Done -> final.pdb, sims/traj.dcd")