import numpy as np

import HamiltonianPy as hp

from DataBase import POINTS, POINTS13, POINTS14, VECTORS


def BasesGenerator(states_index_table, pnum=None, spinz=None):
    assert (pnum is None) or isinstance(pnum, int)
    assert (spinz is None) or isinstance(spinz, (int, float))

    states_num = len(states_index_table)
    if pnum is None:
        return hp.base_vectors(states_num)

    if spinz is None:
        states_indices = list(range(states_num))
        bases = hp.base_vectors([states_indices, pnum])
        bases_h = hp.base_vectors([states_indices, pnum - 1])
        bases_p = hp.base_vectors([states_indices, pnum + 1])
        return bases, bases_h, bases_p

    spin_up_num = (pnum + 2 * spinz) / 2
    spin_down_num = (pnum - 2 * spinz) / 2
    if spin_up_num == int(spin_up_num) and spin_down_num == int(spin_down_num):
        spin_up_num = int(spin_up_num)
        spin_down_num = int(spin_down_num)
    else:
        raise ValueError(
            "The given `pnum` and `spinz` can't be satisfied simultaneously."
        )

    spin_up_states_indices = []
    spin_down_states_indices = []
    for index, state_id in states_index_table:
        if state_id.spin == hp.SPIN_UP:
            spin_up_states_indices.append(index)
        else:
            spin_down_states_indices.append(index)

    bases = hp.base_vectors(
        [spin_up_states_indices, spin_up_num],
        [spin_down_states_indices, spin_down_num],
    )
    bases_h_up = hp.base_vectors(
        [spin_up_states_indices, spin_up_num - 1],
        [spin_down_states_indices, spin_down_num],
    )
    bases_h_down = hp.base_vectors(
        [spin_up_states_indices, spin_up_num],
        [spin_down_states_indices, spin_down_num - 1],
    )
    bases_p_up = hp.base_vectors(
        [spin_up_states_indices, spin_up_num + 1],
        [spin_down_states_indices, spin_down_num],
    )
    bases_p_down = hp.base_vectors(
        [spin_up_states_indices, spin_up_num],
        [spin_down_states_indices, spin_down_num + 1],
    )
    return bases, bases_h_down, bases_h_up, bases_p_down, bases_p_up


cluster = hp.Lattice(POINTS14["(6,7,7,8)"], VECTORS)
states_index_table = hp.IndexTable(
    hp.StateID(point, spin)
    for point in cluster.points for spin in [hp.SPIN_DOWN, hp.SPIN_UP]
)
bases, bases_h_down, bases_h_up, bases_p_down, bases_p_up = BasesGenerator(
    states_index_table, pnum=14, spinz=0
)
print(bases.shape)
print(bases_h_down.shape, bases_h_up.shape)
print(bases_p_down.shape, bases_p_up.shape)
