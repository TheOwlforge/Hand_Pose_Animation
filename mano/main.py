# code mostly taken from original mano model (https://mano.is.tue.mpg.de/),
# but fixed import of .pkl files to work with python3

import numpy as np
import scipy.sparse as sp
import chumpy as ch
from chumpy.ch import MatVecMult
import pickle as pkl
from posemapper import posemap
from verts import verts_core
import json

def ready_arguments(fname_or_dict, posekey4vposed='pose'):

    if not isinstance(fname_or_dict, dict):
        dd = pkl.load(open(fname_or_dict))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert(posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))

    return dd

def load_model(fname_or_dict='MANO_LEFT.pkl', ncomps=6, flat_hand_mean=False, v_template=None):

    if not isinstance(fname_or_dict, dict):
        with open(fname_or_dict, 'rb') as f:
            raw_model_data = pkl.load(f, fix_imports=True, encoding="latin1")
    else:
        raw_model_data = fname_or_dict

    rot = 3  # for global orientation!!!

    hands_components = raw_model_data['hands_components']
    hands_mean       = np.zeros(hands_components.shape[1]) if flat_hand_mean else raw_model_data['hands_mean']

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    pose_coeffs = ch.zeros(rot + selected_components.shape[0])
    full_hand_pose = pose_coeffs[rot:(rot+ncomps)].dot(selected_components)

    raw_model_data['fullpose'] = ch.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))
    raw_model_data['pose'] = pose_coeffs

    Jreg = raw_model_data['J_regressor']
    if not sp.issparse(Jreg):
        raw_model_data['J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    # slightly modify ready_arguments to make sure that it uses the fullpose
    # (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(raw_model_data, posekey4vposed='fullpose')

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }

    result_previous, meta = verts_core(**args)
    result = result_previous + dd['trans'].reshape((1, 3))
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if hasattr(result, 'Jtr'):
        result.J_transformed = result.Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    return result

def save_as_json(modelfile, outputfile):
    # based on https://github.com/YeeCY/SMPLpp/blob/master/SMPL%2B%2B/scripts/preprocess.py
    # but adapted to mano hand model instead of full smpl body model
    with open(modelfile, 'rb') as mf:
        raw_model_data = pkl.load(mf, fix_imports=True, encoding="latin1")

    vertices_template = np.array(raw_model_data['v_template'])

    hands_components = np.array(raw_model_data['hands_components'])
    hands_coefficients = np.array(raw_model_data['hands_coeffs'])
    hands_mean = np.array(raw_model_data['hands_mean'])

    # start from 1 for better compatibility with .obj format
    face_indices = np.array(raw_model_data['f'] + 1)

    weights = np.array(raw_model_data['weights'])

    shape_blend_shapes = np.array(raw_model_data['shapedirs'])
    pose_blend_shapes = np.array(raw_model_data['posedirs'])

    joint_regressor = np.array(raw_model_data['J_regressor'].toarray())
    joints = np.array(raw_model_data['J'])
    kinematic_tree = np.array(raw_model_data['kintree_table'])

    model_data_json = {
        'vertices_template': vertices_template.tolist(),
        'hands_components': hands_components.tolist(),
        'hands_coefficients': hands_coefficients.tolist(),
        'hands_mean': hands_mean.tolist(),
        'face_indices': face_indices.tolist(),
        'weights': weights.tolist(),
        'shape_blend_shapes': shape_blend_shapes.tolist(),
        'pose_blend_shapes': pose_blend_shapes.tolist(),
        'joint_regressor': joint_regressor.tolist(),
        'joints': joints.tolist(),
        'kinematic_tree': kinematic_tree.tolist()
    }

    # wb threw an error
    with open(outputfile, 'w+') as of:
        json.dump(model_data_json, of, indent=4, sort_keys=True)

if __name__ == '__main__':
    m = load_model('model/MANO_RIGHT.pkl', ncomps=6, flat_hand_mean=False)

    # Assign random pose and shape parameters
    m.betas[:] = np.random.rand(m.betas.size) * .03
    #m.pose[:] = np.random.rand(m.pose.size) * 4 - 2
    m.pose[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # m.pose[:3] = [0., 0., 0.]
    # m.pose[3:] = [-0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491]
    # the first 3 elements correspond to global rotation
    # the next ncomps to the hand pose

    # Write to an .obj file
    out_path = './random_hand.obj'
    with open(out_path, 'w') as fp:
        for v in m.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in m.f+1: # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    # convert to json
    save_as_json('model/MANO_RIGHT.pkl', 'model/mano_right.json')
    save_as_json('model/MANO_LEFT.pkl', 'model/mano_left.json')
