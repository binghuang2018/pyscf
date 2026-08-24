#!/usr/bin/env python3
"""Run the MINAO validation with the required heavy-element ECP for iodine."""
from pyscf import gto
import run_validation_fast as rv


def make_mol_with_i_ecp(a, b, r, basis):
    # PySCF MINAO for elements heavier than Kr is a pseudopotential valence
    # basis.  Iodine retains 25 electrons (28-electron core), so the matching
    # def2 ECP must be supplied explicitly.  Without it, RHF sees 53 electrons
    # but only the 13-orbital MINAO valence space.
    ecp = {"I": "def2-svp"} if "I" in (a, b) else {}
    return gto.M(
        atom=[(a, (0, 0, -r / 2)), (b, (0, 0, r / 2))],
        unit="Angstrom",
        basis=basis,
        ecp=ecp,
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


rv.make_mol = make_mol_with_i_ecp

if __name__ == "__main__":
    rv.main()
