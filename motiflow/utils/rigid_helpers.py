from motiflow.utils import rigid_utils as ru


def extract_trans_rots_mat(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats()
    tran = rigid.get_trans()
    return tran, rot


def assemble_rigid_mat(rotmat, trans):
    return ru.Rigid(
        rots=ru.Rotation(rot_mats=rotmat),
        trans=trans,
    )
