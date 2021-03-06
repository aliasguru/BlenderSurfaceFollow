'''
Created on 08.01.2017

@author: renderman
'''
import bpy
import numpy as np

def create_arrows():
    verts = [[ -1.00000000e+00, -1.00000000e+00, 2.96312332e-01],
       [  1.00000000e+00, 1.00000000e+00, 2.96312332e-01],
       [ -1.00000000e+00, 1.00000000e+00, 2.96312302e-01],
       [  1.00000000e+00, -1.00000000e+00, 2.96312302e-01],
       [  7.75208235e-01, -1.00000000e+00, 2.96312302e-01],
       [  1.00000000e+00, 7.75208235e-01, 2.96312302e-01],
       [ -1.00000000e+00, 7.75208235e-01, 2.96312302e-01],
       [  7.75208235e-01, 1.00000000e+00, 2.96312302e-01],
       [ -1.43051147e-06, -5.05597878e+00, 2.96312749e-01],
       [ -9.34385061e-01, -3.44604659e+00, 2.96313107e-01],
       [  1.12072098e+00, -6.54488444e-01, 2.96312302e-01],
       [  3.06722689e+00, 7.75208235e-01, 2.96312302e-01],
       [ -7.75209129e-01, -1.00000048e+00, 2.96312451e-01],
       [  9.34384108e-01, -3.44604588e+00, 2.96312988e-01],
       [ -7.75209665e-01, -3.06722879e+00, 2.96312630e-01],
       [ -1.08122444e+00, -3.43351912e+00, 2.96312660e-01],
       [  6.54478788e-01, -3.11099601e+00, 2.96312481e-01],
       [  3.43351722e+00, 1.08122277e+00, 2.96312302e-01],
       [  3.11099458e+00, -6.54479563e-01, 2.96312243e-01],
       [ -1.43051147e-06, -5.29647636e+00, 2.96312779e-01],
       [ -6.54479742e-01, -3.11099625e+00, 2.96312571e-01],
       [  1.08122325e+00, -3.43351889e+00, 2.96312541e-01],
       [  7.75208950e-01, -3.06722856e+00, 2.96312511e-01],
       [ -6.54487967e-01, -1.12072170e+00, 2.96312451e-01],
       [ -9.99999702e-01, -7.75209367e-01, 2.96312451e-01],
       [ -8.79279017e-01, -8.79279017e-01, 2.96312302e-01],
       [  8.79279017e-01, -8.79279017e-01, 2.96312302e-01],
       [  8.79279017e-01, 8.79279017e-01, 2.96312302e-01],
       [ -8.79279017e-01, 8.79279017e-01, 2.96312302e-01],
       [ -3.06722736e+00, -7.75209188e-01, 2.96312779e-01],
       [  6.54487252e-01, -1.12072098e+00, 2.96312302e-01],
       [  5.29647493e+00, -1.09320899e-06, 2.96312302e-01],
       [  3.11099434e+00, 6.54478490e-01, 2.96312243e-01],
       [  1.12072098e+00, 6.54487252e-01, 2.96312302e-01],
       [ -1.12072098e+00, 6.54487252e-01, 2.96312302e-01],
       [ -3.43351769e+00, -1.08122349e+00, 2.96312839e-01],
       [ -3.11099505e+00, 6.54479265e-01, 2.96312690e-01],
       [  6.54487252e-01, 1.12072098e+00, 2.96312302e-01],
       [ -5.29647493e+00, 9.53674316e-07, 2.96313107e-01],
       [ -3.11099482e+00, -6.54479265e-01, 2.96312720e-01],
       [ -1.12072074e+00, -6.54488325e-01, 2.96312481e-01],
       [ -3.43351769e+00, 1.08122373e+00, 2.96312779e-01],
       [ -3.06722760e+00, 7.75209188e-01, 2.96312749e-01],
       [ -3.44604468e+00, -9.34384346e-01, 2.96313286e-01],
       [  3.43351746e+00, -1.08122420e+00, 2.96312302e-01],
       [  3.06722713e+00, -7.75209427e-01, 2.96312302e-01],
       [  3.44604421e+00, 9.34383690e-01, 2.96312749e-01],
       [ -3.44604516e+00, 9.34384584e-01, 2.96313226e-01],
       [ -5.05597734e+00, 9.53674316e-07, 2.96313047e-01],
       [  1.43051147e-06, 5.05597687e+00, 2.96312600e-01],
       [  9.34385061e-01, 3.44604445e+00, 2.96312869e-01],
       [ -9.34383631e-01, 3.44604397e+00, 2.96313047e-01],
       [  7.75209665e-01, 3.06722689e+00, 2.96312422e-01],
       [  1.00000000e+00, -7.75209427e-01, 2.96312302e-01],
       [  1.08122420e+00, 3.43351722e+00, 2.96312422e-01],
       [ -6.54487491e-01, 1.12072027e+00, 2.96312422e-01],
       [ -6.54478550e-01, 3.11099410e+00, 2.96312481e-01],
       [  1.43051147e-06, 5.29647446e+00, 2.96312630e-01],
       [  6.54479742e-01, 3.11099434e+00, 2.96312362e-01],
       [ -1.08122277e+00, 3.43351698e+00, 2.96312600e-01],
       [ -7.75208473e-01, 3.06722665e+00, 2.96312571e-01],
       [ -7.75208473e-01, 9.99999344e-01, 2.96312422e-01],
       [  3.44604445e+00, -9.34384882e-01, 2.96312749e-01],
       [  5.05597734e+00, -1.09343307e-06, 2.96312302e-01]]

    faces = [[3, 0, 12, 4], [4, 12, 23, 30], [21, 22, 16, 13], [24, 6, 34, 40],
    [14, 12, 23, 20], [45, 53, 10, 18], [22, 4, 30, 16], [0, 3, 26, 25], [3, 1, 27, 26],
    [1, 2, 28, 27], [2, 0, 25, 28], [2, 1, 7, 61], [44, 45, 18, 62], [11, 5, 33, 32], [61, 7, 37, 55],
    [19, 15, 9, 8], [19, 21, 13, 8], [38, 41, 47, 48], [0, 2, 6, 24], [31, 44, 62, 63], [42, 6, 34, 36],
    [57, 54, 50, 49], [15, 14, 20, 9], [29, 24, 40, 39], [41, 42, 36, 47], [38, 35, 43, 48], [57, 59, 51, 49],
    [35, 29, 39, 43], [59, 60, 56, 51], [60, 61, 55, 56], [17, 11, 32, 46], [54, 52, 58, 50], [1, 3, 53, 5], [5, 53, 10, 33],
    [31, 17, 46, 63], [52, 7, 37, 58]]

    arrows = bpy.data.meshes.new("arrows")
    arrows.from_pydata(verts, [], faces)
    arrows.update()

    arrows_ob = bpy.data.objects.new("arrows", arrows)
    bpy.context.scene.objects.link(arrows_ob)
    arrows_ob.show_x_ray = True
    #    smooth = np.ones_like(arrows.polygons)
    #    arrows.polygons.foreach_set('use_smooth', smooth)
    tracker = bpy.data.objects.new('tracker', None)
    bpy.context.scene.objects.link(tracker)
    tracker.scale = [0.001, 0.001, 0.001]
    tracker.location = arrows_ob.location    #    + normal
    constraint = arrows_ob.constraints.new('TRACK_TO')
    constraint.target = tracker
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'

    if 'arrows' not in bpy.data.materials:
        arrows_mat = bpy.data.materials.new('arrows')
        arrows_mat.specular_hardness = 3
        arrows_mat.use_transparency = True
        arrows_mat.alpha = .673
        arrows_mat.use_shadeless = True
    else:
        arrows_mat = bpy.data.materials['arrows']
    if 'arrows_color' not in bpy.data.materials:
        arrows_color = bpy.data.materials.new('arrows_color')
        arrows_color.diffuse_color = [1, 0, 0]
        arrows_color.use_transparency = True
        arrows_color.alpha = .673
        arrows_color.use_shadeless = True
    else:
        arrows_color = bpy.data.materials['arrows_color']

    mats = [arrows_mat, arrows_color]
    for i in mats:
        arrows_ob.data.materials.append(i)

    color_faces = [1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35]
    for i in color_faces:
        arrows.polygons[i].material_index = 1

def create_scale():
    verts = [[-0.34357, 0.52956, 1.07163],
       [-0.52381, 0.7043 , 1.07163],
       [-0.     , 1.60682, 1.07163],
       [ 0.52381, 0.7043 , 1.07163],
       [ 0.34357, 0.52956, 1.07163],
       [ 0.28077, 0.68454, 1.07163],
       [ 0.32896, 0.73126, 1.07163],
       [-0.     , 1.29806, 1.07163],
       [-0.32896, 0.73126, 1.07163],
       [-0.28077, 0.68454, 1.07163],
       [-0.44664, 1.60682, 1.39312],
       [-0.68095, 1.83399, 1.39312],
       [-0.     , 3.00727, 1.39312],
       [ 0.68095, 1.83399, 1.39312],
       [ 0.44664, 1.60682, 1.39312],
       [ 0.365  , 1.80831, 1.39312],
       [ 0.42765, 1.86904, 1.39312],
       [-0.     , 2.60588, 1.39312],
       [-0.42765, 1.86904, 1.39312],
       [-0.36501, 1.80831, 1.39312],
       [-0.58064, 3.00727, 1.81106],
       [-0.88523, 3.30259, 1.81106],
       [-0.     , 4.82785, 1.81106],
       [ 0.88523, 3.30259, 1.81106],
       [ 0.58063, 3.00727, 1.81106],
       [ 0.47451, 3.2692 , 1.81106],
       [ 0.55594, 3.34815, 1.81106],
       [-0.     , 4.30605, 1.81106],
       [-0.55594, 3.34815, 1.81106],
       [-0.47451, 3.2692 , 1.81106],
       [ 0.52956, 0.34357, 1.07163],
       [ 0.7043 , 0.52381, 1.07163],
       [ 1.60682, 0.     , 1.07163],
       [ 0.7043 , -0.52381, 1.07163],
       [ 0.52956, -0.34357, 1.07163],
       [ 0.68454, -0.28077, 1.07163],
       [ 0.73126, -0.32896, 1.07163],
       [ 1.29806, 0.     , 1.07163],
       [ 0.73126, 0.32896, 1.07163],
       [ 0.68454, 0.28077, 1.07163],
       [ 1.60682, 0.44664, 1.39312],
       [ 1.83399, 0.68095, 1.39312],
       [ 3.00727, 0.     , 1.39312],
       [ 1.83399, -0.68095, 1.39312],
       [ 1.60682, -0.44664, 1.39312],
       [ 1.80831, -0.365  , 1.39312],
       [ 1.86904, -0.42765, 1.39312],
       [ 2.60588, 0.     , 1.39312],
       [ 1.86904, 0.42765, 1.39312],
       [ 1.80831, 0.36501, 1.39312],
       [ 3.00727, 0.58064, 1.81106],
       [ 3.30259, 0.88523, 1.81106],
       [ 4.82785, 0.     , 1.81106],
       [ 3.30259, -0.88523, 1.81106],
       [ 3.00727, -0.58063, 1.81106],
       [ 3.2692 , -0.47451, 1.81106],
       [ 3.34815, -0.55594, 1.81106],
       [ 4.30605, 0.     , 1.81106],
       [ 3.34815, 0.55595, 1.81106],
       [ 3.2692 , 0.47451, 1.81106],
       [ 0.34357, -0.52956, 1.07163],
       [ 0.52381, -0.7043 , 1.07163],
       [ 0.     , -1.60682, 1.07163],
       [-0.52381, -0.7043 , 1.07163],
       [-0.34357, -0.52956, 1.07163],
       [-0.28077, -0.68454, 1.07163],
       [-0.32896, -0.73126, 1.07163],
       [ 0.     , -1.29806, 1.07163],
       [ 0.32896, -0.73126, 1.07163],
       [ 0.28077, -0.68454, 1.07163],
       [ 0.44664, -1.60682, 1.39312],
       [ 0.68095, -1.83399, 1.39312],
       [ 0.     , -3.00727, 1.39312],
       [-0.68095, -1.83399, 1.39312],
       [-0.44664, -1.60682, 1.39312],
       [-0.365  , -1.80831, 1.39312],
       [-0.42765, -1.86904, 1.39312],
       [ 0.     , -2.60588, 1.39312],
       [ 0.42765, -1.86904, 1.39312],
       [ 0.36501, -1.80831, 1.39312],
       [ 0.58064, -3.00727, 1.81106],
       [ 0.88523, -3.30259, 1.81106],
       [ 0.     , -4.82785, 1.81106],
       [-0.88523, -3.30259, 1.81106],
       [-0.58063, -3.00727, 1.81106],
       [-0.47451, -3.2692 , 1.81106],
       [-0.55594, -3.34815, 1.81106],
       [ 0.     , -4.30605, 1.81106],
       [ 0.55595, -3.34815, 1.81106],
       [ 0.47451, -3.2692 , 1.81106],
       [-0.52956, -0.34357, 1.07163],
       [-0.7043 , -0.52381, 1.07163],
       [-1.60682, -0.     , 1.07163],
       [-0.7043 , 0.52381, 1.07163],
       [-0.52956, 0.34357, 1.07163],
       [-0.68454, 0.28077, 1.07163],
       [-0.73126, 0.32896, 1.07163],
       [-1.29806, -0.     , 1.07163],
       [-0.73126, -0.32896, 1.07163],
       [-0.68454, -0.28077, 1.07163],
       [-1.60682, -0.44664, 1.39312],
       [-1.83399, -0.68095, 1.39312],
       [-3.00727, -0.     , 1.39312],
       [-1.83399, 0.68095, 1.39312],
       [-1.60682, 0.44664, 1.39312],
       [-1.80831, 0.365  , 1.39312],
       [-1.86904, 0.42765, 1.39312],
       [-2.60588, -0.     , 1.39312],
       [-1.86904, -0.42765, 1.39312],
       [-1.80831, -0.36501, 1.39312],
       [-3.00727, -0.58064, 1.81106],
       [-3.30259, -0.88523, 1.81106],
       [-4.82785, -0.     , 1.81106],
       [-3.30259, 0.88523, 1.81106],
       [-3.00727, 0.58063, 1.81106],
       [-3.2692 , 0.47451, 1.81106],
       [-3.34815, 0.55594, 1.81106],
       [-4.30605, -0.     , 1.81106],
       [-3.34815, -0.55595, 1.81106],
       [-3.2692 , -0.47451, 1.81106]]

    faces = [[6, 7, 8, 9, 5], [6, 5, 4, 3], [8, 7, 2, 1], [5, 9, 0, 4], [7, 6, 3, 2], [9, 8, 1, 0],
     [16, 17, 18, 19, 15], [16, 15, 14, 13], [18, 17, 12, 11], [15, 19, 10, 14], [17, 16, 13, 12],
     [19, 18, 11, 10], [26, 27, 28, 29, 25], [26, 25, 24, 23], [28, 27, 22, 21], [25, 29, 20, 24],
     [27, 26, 23, 22], [29, 28, 21, 20], [36, 37, 38, 39, 35], [36, 35, 34, 33], [38, 37, 32, 31],
     [35, 39, 30, 34], [37, 36, 33, 32], [39, 38, 31, 30], [46, 47, 48, 49, 45], [46, 45, 44, 43],
     [48, 47, 42, 41], [45, 49, 40, 44], [47, 46, 43, 42], [49, 48, 41, 40], [56, 57, 58, 59, 55],
     [56, 55, 54, 53], [58, 57, 52, 51], [55, 59, 50, 54], [57, 56, 53, 52], [59, 58, 51, 50],
     [66, 67, 68, 69, 65], [66, 65, 64, 63], [68, 67, 62, 61], [65, 69, 60, 64], [67, 66, 63, 62],
     [69, 68, 61, 60], [76, 77, 78, 79, 75], [76, 75, 74, 73], [78, 77, 72, 71], [75, 79, 70, 74],
     [77, 76, 73, 72], [79, 78, 71, 70], [86, 87, 88, 89, 85], [86, 85, 84, 83], [88, 87, 82, 81],
     [85, 89, 80, 84], [87, 86, 83, 82], [89, 88, 81, 80], [96, 97, 98, 99, 95], [96, 95, 94, 93],
     [98, 97, 92, 91], [95, 99, 90, 94], [97, 96, 93, 92], [99, 98, 91, 90], [106, 107, 108, 109, 105],
     [106, 105, 104, 103], [108, 107, 102, 101], [105, 109, 100, 104], [107, 106, 103, 102],
     [109, 108, 101, 100], [116, 117, 118, 119, 115], [116, 115, 114, 113], [118, 117, 112, 111],
     [115, 119, 110, 114], [117, 116, 113, 112], [119, 118, 111, 110]]

    scale = bpy.data.meshes.new("scale")
    scale.from_pydata(verts, [], faces)
    scale.update()

    scale_ob = bpy.data.objects.new("scale", scale)
    bpy.context.scene.objects.link(scale_ob)
    scale_ob.show_x_ray = True
    tracker = bpy.data.objects.new('tracker', None)
    bpy.context.scene.objects.link(tracker)
    tracker.scale = [0.001, 0.001, 0.001]
    tracker.location = scale_ob.location    #    + normal
    constraint = scale_ob.constraints.new('TRACK_TO')
    constraint.target = tracker
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'

    if 'arrows' not in bpy.data.materials:
        arrows_mat = bpy.data.materials.new('arrows')
        arrows_mat.specular_hardness = 3
        arrows_mat.use_transparency = True
        arrows_mat.alpha = .3
        arrows_mat.use_shadeless = True
    else:
        arrows_mat = bpy.data.materials['arrows']
    if 'arrows_color' not in bpy.data.materials:
        arrows_color = bpy.data.materials.new('arrows_color')
        arrows_color.diffuse_color = [1, 0, 0]
        arrows_color.use_transparency = True
        arrows_color.alpha = .673
        arrows_color.use_shadeless = True
    else:
        arrows_color = bpy.data.materials['arrows_color']

    mats = [arrows_mat, arrows_color]
    for i in mats:
        scale_ob.data.materials.append(i)

    color_faces = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25,
    26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52,
    53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71]

    for i in color_faces:
        scale.polygons[i].material_index = 1

def create_rotate_0():
    verts = [[ 0.46297559, -1.09622312, 0.        ],
       [ 0.44762817, -1.17309833, 0.        ],
       [-0.31693506, -1.20904827, 0.        ],
       [ 0.0311211 , -0.5273543 , 0.        ],
       [ 0.10748389, -0.5450756 , 0.        ],
       [ 0.21983896, -0.3853251 , 0.        ],
       [-0.06577085, -0.31904328, 0.        ],
       [-0.61998093, -1.40451419, 0.        ],
       [ 0.59744227, -1.34727466, 0.        ],
       [ 0.65484381, -1.05974841, 0.        ],
       [ 0.74123406, -0.63505244, 0.        ],
       [ 0.53988403, -0.45892119, 0.        ],
       [ 0.45764941, -0.54064721, 0.        ],
       [ 0.63252699, -0.74308652, 0.        ],
       [ 0.34447849, -0.91250736, 0.        ],
       [ 0.22598176, -0.72879165, 0.        ],
       [ 0.50984234, -0.83494037, 0.        ],
       [ 0.36484101, -0.61013311, 0.        ],
       [ 0.83384776, -0.51294047, 0.        ],
       [ 0.60994452, -0.36654601, 0.        ],
       [ 0.66646707, -0.26531908, 0.        ],
       [ 0.7347821 , -0.04432551, 0.        ],
       [ 0.99887198, -0.08699386, 0.        ],
       [ 0.70835137, -0.15721081, 0.        ],
       [ 0.96393251, -0.2362179 , 0.        ],
       [ 0.90856528, -0.37912741, 0.        ],
       [ 1.01270318, 0.0656408 , 0.        ],
       [ 0.74524528, 0.07113946, 0.        ],
       [ 1.00515699, 0.21871488, 0.        ],
       [ 0.73953694, 0.18693718, 0.        ],
       [-0.46297571, 1.09622312, 0.        ],
       [-0.44762829, 1.17309833, 0.        ],
       [ 0.31693497, 1.20904827, 0.        ],
       [-0.03112114, 0.5273543 , 0.        ],
       [-0.10748394, 0.5450756 , 0.        ],
       [-0.21983901, 0.3853251 , 0.        ],
       [ 0.06577083, 0.31904328, 0.        ],
       [ 0.61998075, 1.40451419, 0.        ],
       [-0.59744239, 1.34727466, 0.        ],
       [-0.65484393, 1.05974841, 0.        ],
       [-0.74123406, 0.63505244, 0.        ],
       [-0.53988403, 0.4589211 , 0.        ],
       [-0.45764941, 0.54064709, 0.        ],
       [-0.63252699, 0.7430864 , 0.        ],
       [-0.34447852, 0.91250736, 0.        ],
       [-0.2259818 , 0.72879165, 0.        ],
       [-0.5098424 , 0.83494037, 0.        ],
       [-0.36484107, 0.61013311, 0.        ],
       [-0.83384776, 0.51294035, 0.        ],
       [-0.60994452, 0.36654595, 0.        ],
       [-0.66646707, 0.26531902, 0.        ],
       [-0.7347821 , 0.04432545, 0.        ],
       [-0.99887198, 0.08699376, 0.        ],
       [-0.70835137, 0.15721075, 0.        ],
       [-0.96393251, 0.2362178 , 0.        ],
       [-0.90856528, 0.37912732, 0.        ],
       [-1.01270318, -0.0656409 , 0.        ],
       [-0.74524528, -0.07113954, 0.        ],
       [-1.00515699, -0.21871497, 0.        ],
       [-0.73953694, -0.18693726, 0.        ]]

    faces = [[4, 15, 17, 5], [2, 3, 6, 7], [17, 16, 13, 12], [3, 4, 5, 6], [14, 0, 9, 16],
     [1, 2, 7, 8], [11, 10, 18, 19], [1, 0, 14, 15, 4, 3, 2], [12, 13, 10, 11],
      [15, 14, 16, 17], [0, 1, 8, 9], [21, 22, 26, 27], [23, 24, 22, 21], [27, 26, 28, 29],
       [20, 25, 24, 23], [19, 18, 25, 20], [34, 45, 47, 35], [32, 33, 36, 37], [47, 46, 43, 42],
        [33, 34, 35, 36], [44, 30, 39, 46], [31, 32, 37, 38], [41, 40, 48, 49],
         [31, 30, 44, 45, 34, 33, 32], [42, 43, 40, 41], [45, 44, 46, 47], [30, 31, 38, 39],
          [51, 52, 56, 57], [53, 54, 52, 51], [57, 56, 58, 59], [50, 55, 54, 53], [49, 48, 55, 50]]

    rotate_0 = bpy.data.meshes.new("rotate_0")
    rotate_0.from_pydata(verts, [], faces)
    rotate_0.update()

    rotate_0 = bpy.data.objects.new("rotate_0", rotate_0)
    bpy.context.scene.objects.link(rotate_0)
    rotate_0.show_x_ray = True
    #    smooth = np.ones_like(arrows.polygons)
    #    arrows.polygons.foreach_set('use_smooth', smooth)
    tracker_0 = bpy.data.objects.new('tracker_0', None)
    bpy.context.scene.objects.link(tracker_0)
    tracker_0.scale = [0.001, 0.001, 0.001]
    tracker_0.location = rotate_0.location    #    + normal
    constraint = rotate_0.constraints.new('TRACK_TO')
    constraint.target = tracker_0
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'

    if 'rotate' not in bpy.data.materials:
        rotate_mat = bpy.data.materials.new('rotate')
        rotate_mat.specular_hardness = 3
        rotate_mat.use_transparency = True
        rotate_mat.alpha = .673
        rotate_mat.use_shadeless = True
    else:
        rotate_mat = bpy.data.materials['rotate']
    if 'rotate_color' not in bpy.data.materials:
        rotate_color = bpy.data.materials.new('rotate_color')
        rotate_color.diffuse_color = [1, 0, 0]
        rotate_color.use_transparency = True
        rotate_color.alpha = .673
        rotate_color.use_shadeless = True
    else:
        rotate_color = bpy.data.materials['rotate_color']

    mats = [rotate_mat, rotate_color]
    for i in mats:
        rotate_0.data.materials.append(i)

    color_faces = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,
     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
    for i in color_faces:
        rotate_0.data.polygons[i].material_index = 1

def create_rotate_1():
    verts = [[  4.50903296e-01, -4.99970317e-02, 0.00000000e+00],
       [  4.49821234e-01, 4.99970913e-02, 0.00000000e+00],
       [  5.99176466e-01, -4.99970317e-02, 0.00000000e+00],
       [  5.98094404e-01, 4.99970913e-02, 0.00000000e+00],
       [  6.00030363e-01, -1.28906727e-01, 0.00000000e+00],
       [  5.97240508e-01, 1.28906786e-01, 0.00000000e+00],
       [  6.58333242e-01, -1.81031227e-01, 0.00000000e+00],
       [  6.54415190e-01, 1.81031287e-01, 0.00000000e+00],
       [  9.48275685e-01, 2.98023224e-08, 0.00000000e+00],
       [  0.00000000e+00, 4.37937856e-01, 0.00000000e+00],
       [ -8.54374319e-02, 4.29522991e-01, 0.00000000e+00],
       [ -1.67591557e-01, 4.04601812e-01, 0.00000000e+00],
       [ -2.43305206e-01, 3.64132047e-01, 0.00000000e+00],
       [ -3.09668809e-01, 3.09668899e-01, 0.00000000e+00],
       [ -3.64132017e-01, 2.43305266e-01, 0.00000000e+00],
       [ -4.04601812e-01, 1.67591572e-01, 0.00000000e+00],
       [ -4.29522991e-01, 8.54374766e-02, 0.00000000e+00],
       [ -4.37937856e-01, 5.96046448e-08, 0.00000000e+00],
       [ -4.29522991e-01, -8.54373574e-02, 0.00000000e+00],
       [ -4.04601812e-01, -1.67591453e-01, 0.00000000e+00],
       [ -3.64132017e-01, -2.43305176e-01, 0.00000000e+00],
       [ -3.09668809e-01, -3.09668779e-01, 0.00000000e+00],
       [ -2.43305206e-01, -3.64131987e-01, 0.00000000e+00],
       [ -1.67591482e-01, -4.04601812e-01, 0.00000000e+00],
       [ -8.54373276e-02, -4.29522932e-01, 0.00000000e+00],
       [  1.42698255e-07, -4.37937796e-01, 0.00000000e+00],
       [  8.54376107e-02, -4.29522872e-01, 0.00000000e+00],
       [  1.67591751e-01, -4.04601634e-01, 0.00000000e+00],
       [  2.43305445e-01, -3.64131838e-01, 0.00000000e+00],
       [  3.09669018e-01, -3.09668601e-01, 0.00000000e+00],
       [  3.64132166e-01, -2.43304938e-01, 0.00000000e+00],
       [  4.04601932e-01, -1.67591184e-01, 0.00000000e+00],
       [  4.29523051e-01, -8.54370296e-02, 0.00000000e+00],
       [  4.37937856e-01, 4.47034836e-07, 0.00000000e+00],
       [  4.29522872e-01, 8.54378939e-02, 0.00000000e+00],
       [  4.04601634e-01, 1.67592019e-01, 0.00000000e+00],
       [  3.64131689e-01, 2.43305743e-01, 0.00000000e+00],
       [  3.09668422e-01, 3.09669256e-01, 0.00000000e+00],
       [  2.43304729e-01, 3.64132345e-01, 0.00000000e+00],
       [  1.67590961e-01, 4.04602051e-01, 0.00000000e+00],
       [  8.54367763e-02, 4.29523110e-01, 0.00000000e+00],
       [  7.23102540e-02, 3.63531113e-01, 0.00000000e+00],
       [  1.41842246e-01, 3.42438936e-01, 0.00000000e+00],
       [  2.05923334e-01, 3.08187008e-01, 0.00000000e+00],
       [  2.62090892e-01, 2.62091637e-01, 0.00000000e+00],
       [  3.08186412e-01, 2.05924213e-01, 0.00000000e+00],
       [  3.42438549e-01, 1.41843140e-01, 0.00000000e+00],
       [  3.63530874e-01, 7.23112226e-02, 0.00000000e+00],
       [  3.70653003e-01, 3.87430191e-07, 0.00000000e+00],
       [  3.63531053e-01, -7.23104477e-02, 0.00000000e+00],
       [  3.42438817e-01, -1.41842425e-01, 0.00000000e+00],
       [  3.08186829e-01, -2.05923513e-01, 0.00000000e+00],
       [  2.62091398e-01, -2.62091041e-01, 0.00000000e+00],
       [  2.05923945e-01, -3.08186531e-01, 0.00000000e+00],
       [  1.41842917e-01, -3.42438549e-01, 0.00000000e+00],
       [  7.23109618e-02, -3.63530874e-01, 0.00000000e+00],
       [  1.20774061e-07, -3.70652944e-01, 0.00000000e+00],
       [ -7.23107159e-02, -3.63530934e-01, 0.00000000e+00],
       [ -1.41842693e-01, -3.42438698e-01, 0.00000000e+00],
       [ -2.05923736e-01, -3.08186650e-01, 0.00000000e+00],
       [ -2.62091219e-01, -2.62091190e-01, 0.00000000e+00],
       [ -3.08186710e-01, -2.05923706e-01, 0.00000000e+00],
       [ -3.42438698e-01, -1.41842663e-01, 0.00000000e+00],
       [ -3.63530993e-01, -7.23107457e-02, 0.00000000e+00],
       [ -3.70653003e-01, 5.96046448e-08, 0.00000000e+00],
       [ -3.63530993e-01, 7.23108649e-02, 0.00000000e+00],
       [ -3.42438698e-01, 1.41842782e-01, 0.00000000e+00],
       [ -3.08186710e-01, 2.05923796e-01, 0.00000000e+00],
       [ -2.62091219e-01, 2.62091279e-01, 0.00000000e+00],
       [ -2.05923736e-01, 3.08186769e-01, 0.00000000e+00],
       [ -1.41842753e-01, 3.42438698e-01, 0.00000000e+00],
       [ -7.23108053e-02, 3.63530993e-01, 0.00000000e+00],
       [  0.00000000e+00, 3.70653033e-01, 0.00000000e+00],
       [  2.83483475e-01, -3.54190022e-01, 0.00000000e+00],
       [  3.53424877e-01, -2.82718390e-01, 0.00000000e+00],
       [  3.88328433e-01, -4.59035009e-01, 0.00000000e+00],
       [  4.58269835e-01, -3.87563378e-01, 0.00000000e+00],
       [  3.33134651e-01, -5.15436411e-01, 0.00000000e+00],
       [  5.13463616e-01, -3.31162006e-01, 0.00000000e+00],
       [  3.37503433e-01, -5.93520284e-01, 0.00000000e+00],
       [  5.90749800e-01, -3.34733069e-01, 0.00000000e+00],
       [  6.70532107e-01, -6.70532227e-01, 0.00000000e+00],
       [ -4.99971211e-02, -4.50903237e-01, 0.00000000e+00],
       [  4.99969721e-02, -4.49821204e-01, 0.00000000e+00],
       [ -4.99971807e-02, -5.99176407e-01, 0.00000000e+00],
       [  4.99969423e-02, -5.98094344e-01, 0.00000000e+00],
       [ -1.28906846e-01, -6.00030303e-01, 0.00000000e+00],
       [  1.28906608e-01, -5.97240448e-01, 0.00000000e+00],
       [ -1.81031346e-01, -6.58333182e-01, 0.00000000e+00],
       [  1.81031108e-01, -6.54415250e-01, 0.00000000e+00],
       [ -1.19209290e-07, -9.48275685e-01, 0.00000000e+00],
       [ -3.54190081e-01, -2.83483386e-01, 0.00000000e+00],
       [ -2.82718480e-01, -3.53424758e-01, 0.00000000e+00],
       [ -4.59035069e-01, -3.88328284e-01, 0.00000000e+00],
       [ -3.87563467e-01, -4.58269715e-01, 0.00000000e+00],
       [ -5.15436411e-01, -3.33134502e-01, 0.00000000e+00],
       [ -3.31162125e-01, -5.13463497e-01, 0.00000000e+00],
       [ -5.93520403e-01, -3.37503284e-01, 0.00000000e+00],
       [ -3.34733188e-01, -5.90749741e-01, 0.00000000e+00],
       [ -6.70532286e-01, -6.70531988e-01, 0.00000000e+00],
       [ -4.50903237e-01, 4.99972105e-02, 0.00000000e+00],
       [ -4.49821234e-01, -4.99968529e-02, 0.00000000e+00],
       [ -5.99176407e-01, 4.99972999e-02, 0.00000000e+00],
       [ -5.98094344e-01, -4.99967933e-02, 0.00000000e+00],
       [ -6.00030243e-01, 1.28906965e-01, 0.00000000e+00],
       [ -5.97240508e-01, -1.28906459e-01, 0.00000000e+00],
       [ -6.58333182e-01, 1.81031495e-01, 0.00000000e+00],
       [ -6.54415250e-01, -1.81030959e-01, 0.00000000e+00],
       [ -9.48275685e-01, 2.68220901e-07, 0.00000000e+00],
       [ -2.83483386e-01, 3.54190111e-01, 0.00000000e+00],
       [ -3.53424728e-01, 2.82718539e-01, 0.00000000e+00],
       [ -3.88328224e-01, 4.59035158e-01, 0.00000000e+00],
       [ -4.58269626e-01, 3.87563527e-01, 0.00000000e+00],
       [ -3.33134443e-01, 5.15436411e-01, 0.00000000e+00],
       [ -5.13463438e-01, 3.31162214e-01, 0.00000000e+00],
       [ -3.37503195e-01, 5.93520522e-01, 0.00000000e+00],
       [ -5.90749621e-01, 3.34733248e-01, 0.00000000e+00],
       [ -6.70531929e-01, 6.70532465e-01, 0.00000000e+00],
       [  4.99972105e-02, 4.50903296e-01, 0.00000000e+00],
       [ -4.99967933e-02, 4.49821234e-01, 0.00000000e+00],
       [  4.99973893e-02, 5.99176407e-01, 0.00000000e+00],
       [ -4.99967337e-02, 5.98094344e-01, 0.00000000e+00],
       [  1.28906995e-01, 6.00030184e-01, 0.00000000e+00],
       [ -1.28906399e-01, 5.97240567e-01, 0.00000000e+00],
       [  1.81031644e-01, 6.58333182e-01, 0.00000000e+00],
       [ -1.81030869e-01, 6.54415250e-01, 0.00000000e+00],
       [  4.17232513e-07, 9.48275685e-01, 0.00000000e+00],
       [  3.54190141e-01, 2.83483386e-01, 0.00000000e+00],
       [  2.82718569e-01, 3.53424668e-01, 0.00000000e+00],
       [  4.59035218e-01, 3.88328195e-01, 0.00000000e+00],
       [  3.87563556e-01, 4.58269596e-01, 0.00000000e+00],
       [  5.15436411e-01, 3.33134413e-01, 0.00000000e+00],
       [  3.31162304e-01, 5.13463497e-01, 0.00000000e+00],
       [  5.93520582e-01, 3.37503076e-01, 0.00000000e+00],
       [  3.34733367e-01, 5.90749621e-01, 0.00000000e+00],
       [  6.70532525e-01, 6.70531869e-01, 0.00000000e+00],
       [  5.01745939e-02, 2.52246737e-01, 0.00000000e+00],
       [  9.84214097e-02, 2.37611309e-01, 0.00000000e+00],
       [  1.42885953e-01, 2.13844612e-01, 0.00000000e+00],
       [  2.07520068e-01, -4.12780717e-02, 0.00000000e+00],
       [  2.07519948e-01, 4.12785523e-02, 0.00000000e+00],
       [  1.17550403e-01, 1.75927177e-01, 0.00000000e+00],
       [  2.52246559e-01, 5.01752794e-02, 0.00000000e+00],
       [  2.57188469e-01, 2.82514463e-07, 0.00000000e+00],
       [  2.52246708e-01, -5.01747131e-02, 0.00000000e+00],
       [  2.37611234e-01, -9.84215215e-02, 0.00000000e+00],
       [  2.13844478e-01, -1.42886057e-01, 0.00000000e+00],
       [  1.81859806e-01, -1.81859568e-01, 0.00000000e+00],
       [  1.42886370e-01, -2.13844270e-01, 0.00000000e+00],
       [  9.84218717e-02, -2.37611026e-01, 0.00000000e+00],
       [  5.01750857e-02, -2.52246559e-01, 0.00000000e+00],
       [  8.38026324e-08, -2.57188380e-01, 0.00000000e+00],
       [ -5.01749143e-02, -2.52246618e-01, 0.00000000e+00],
       [ -9.84217152e-02, -2.37611145e-01, 0.00000000e+00],
       [ -1.42886236e-01, -2.13844359e-01, 0.00000000e+00],
       [  2.11585581e-01, 2.40347447e-07, 0.00000000e+00],
       [  8.09700042e-02, 1.95479721e-01, 0.00000000e+00],
       [  4.12779823e-02, 2.07520083e-01, 0.00000000e+00],
       [ -2.52246648e-01, -5.01749218e-02, 0.00000000e+00],
       [ -2.57188469e-01, 5.50430919e-08, 0.00000000e+00],
       [ -2.52246648e-01, 5.01750298e-02, 0.00000000e+00],
       [ -2.37611145e-01, 9.84217972e-02, 0.00000000e+00],
       [ -2.13844404e-01, 1.42886281e-01, 0.00000000e+00],
       [ -1.81859687e-01, 1.81859732e-01, 0.00000000e+00],
       [ -1.42886236e-01, 2.13844448e-01, 0.00000000e+00],
       [ -9.84217599e-02, 2.37611145e-01, 0.00000000e+00],
       [ -5.01749739e-02, 2.52246678e-01, 0.00000000e+00],
       [  0.00000000e+00, 2.57188499e-01, 0.00000000e+00],
       [  1.95479646e-01, -8.09700862e-02, 0.00000000e+00],
       [  1.75927058e-01, -1.17550485e-01, 0.00000000e+00],
       [  1.49613678e-01, -1.49613485e-01, 0.00000000e+00],
       [  1.17550746e-01, -1.75926879e-01, 0.00000000e+00],
       [  8.09703842e-02, -1.95479468e-01, 0.00000000e+00],
       [  4.12783846e-02, -2.07519948e-01, 0.00000000e+00],
       [  6.89433293e-08, -2.11585522e-01, 0.00000000e+00],
       [ -4.12782468e-02, -2.07519993e-01, 0.00000000e+00],
       [ -8.09702575e-02, -1.95479572e-01, 0.00000000e+00],
       [ -1.17550634e-01, -1.75926954e-01, 0.00000000e+00],
       [ -2.07520023e-01, -4.12782431e-02, 0.00000000e+00],
       [ -2.11585581e-01, 5.32097459e-08, 0.00000000e+00],
       [ -2.07520023e-01, 4.12783474e-02, 0.00000000e+00],
       [ -1.95479572e-01, 8.09703320e-02, 0.00000000e+00],
       [ -1.75926998e-01, 1.17550679e-01, 0.00000000e+00],
       [ -1.49613589e-01, 1.49613634e-01, 0.00000000e+00],
       [ -1.17550634e-01, 1.75927043e-01, 0.00000000e+00],
       [ -8.09702948e-02, 1.95479587e-01, 0.00000000e+00],
       [ -4.12782952e-02, 2.07520038e-01, 0.00000000e+00],
       [  0.00000000e+00, 2.11585611e-01, 0.00000000e+00],
       [  1.50794297e-01, 4.98524271e-02, 0.00000000e+00],
       [  2.94287562e-01, 9.72911119e-02, 0.00000000e+00],
       [ -2.94287711e-01, -9.72907022e-02, 0.00000000e+00],
       [ -1.50794327e-01, -4.98521701e-02, 0.00000000e+00],
       [  1.65383726e-01, 1.66088805e-01, 0.00000000e+00],
       [ -1.65383920e-01, -1.66088536e-01, 0.00000000e+00]]

    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [6, 7, 8], [30, 29, 52, 51], [23, 22, 59, 58], [38, 37, 44, 43],
    [16, 15, 66, 65], [36, 35, 46, 45], [11, 10, 71, 70], [29, 28, 53, 52], [22, 21, 60, 59], [24, 23, 58, 57],
    [15, 14, 67, 66], [35, 34, 47, 46], [10, 9, 72, 71], [28, 27, 54, 53], [21, 20, 61, 60], [9, 40, 41, 72],
    [34, 33, 48, 47], [14, 13, 68, 67], [20, 19, 62, 61], [27, 26, 55, 54], [33, 32, 49, 48], [40, 39, 42, 41],
    [19, 18, 63, 62], [13, 12, 69, 68], [32, 31, 50, 49], [26, 25, 56, 55], [18, 17, 64, 63], [39, 38, 43, 42],
    [31, 30, 51, 50], [12, 11, 70, 69], [17, 16, 65, 64], [37, 36, 45, 44], [25, 24, 57, 56], [73, 74, 76, 75],
    [77, 78, 80, 79], [79, 80, 81], [82, 83, 85, 84], [86, 87, 89, 88], [88, 89, 90], [91, 92, 94, 93],
    [95, 96, 98, 97], [97, 98, 99], [100, 101, 103, 102], [104, 105, 107, 106], [106, 107, 108],
    [109, 110, 112, 111], [113, 114, 116, 115], [115, 116, 117], [118, 119, 121, 120], [122, 123, 125, 124],
    [124, 125, 126], [127, 128, 130, 129], [131, 132, 134, 133], [133, 134, 135], [144, 145, 168, 139],
    [165, 166, 186, 185], [160, 161, 181, 180], [151, 152, 175, 174], [146, 147, 170, 169], [162, 163, 183, 182],
    [153, 154, 177, 176], [148, 149, 172, 171], [166, 167, 187, 186], [164, 165, 185, 184], [150, 151, 174, 173],
    [136, 137, 156, 157], [167, 136, 157, 187], [152, 153, 176, 175], [143, 144, 139, 155], [161, 162, 182, 181],
    [159, 160, 180, 179], [145, 146, 169, 168], [163, 164, 184, 183], [147, 148, 171, 170], [142, 143, 155, 140],
    [137, 138, 141, 156], [158, 159, 179, 178], [149, 150, 173, 172], [158, 178, 191, 190], [142, 140, 188, 189],
    [190, 191, 193], [189, 188, 192]]

    rotate_1 = bpy.data.meshes.new("rotate_1")
    rotate_1.from_pydata(verts, [], faces)
    rotate_1.update()

    rotate_1 = bpy.data.objects.new("rotate_1", rotate_1)
    rotate_1.scale = np.array([.1, .1, .1])
    bpy.context.scene.objects.link(rotate_1)
    rotate_1.show_x_ray = True
    #    smooth = np.ones_like(arrows.polygons)
    #    arrows.polygons.foreach_set('use_smooth', smooth)
    tracker_1 = bpy.data.objects.new('tracker_1', None)
    bpy.context.scene.objects.link(tracker_1)
    tracker_1.scale = [0.001, 0.001, 0.001]
    tracker_1.location = rotate_1.location    #    + normal
    constraint = rotate_1.constraints.new('TRACK_TO')
    constraint.target = tracker_1
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'

    if 'rotate' not in bpy.data.materials:
        rotate_mat = bpy.data.materials.new('rotate')
        rotate_mat.specular_hardness = 3
        rotate_mat.use_transparency = True
        rotate_mat.alpha = .673
        rotate_mat.use_shadeless = True
    else:
        rotate_mat = bpy.data.materials['rotate']
    if 'rotate_color' not in bpy.data.materials:
        rotate_color = bpy.data.materials.new('rotate_color')
        rotate_color.diffuse_color = [1, 0, 0]
        rotate_color.use_transparency = True
        rotate_color.alpha = .673
        rotate_color.use_shadeless = True
    else:
        rotate_color = bpy.data.materials['rotate_color']

    mats = [rotate_mat, rotate_color]
    for i in mats:
        rotate_1.data.materials.append(i)

    color_faces = [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
    for i in color_faces:
        rotate_1.data.polygons[i].material_index = 1

def create_rotate_2():
    verts = [[  7.94371665e-02, 3.99360478e-01, 0.00000000e+00],
       [  1.55822203e-01, 3.76189440e-01, 0.00000000e+00],
       [  2.26219088e-01, 3.38561654e-01, 0.00000000e+00],
       [  3.28548729e-01, -6.53521642e-02, 0.00000000e+00],
       [  3.28548551e-01, 6.53526485e-02, 0.00000000e+00],
       [  1.86107501e-01, 2.78530240e-01, 0.00000000e+00],
       [  3.99360359e-01, 7.94380754e-02, 0.00000000e+00],
       [  4.07184482e-01, 3.08231790e-07, 0.00000000e+00],
       [  3.99360597e-01, -7.94374496e-02, 0.00000000e+00],
       [  3.76189470e-01, -1.55822471e-01, 0.00000000e+00],
       [  3.38561594e-01, -2.26219371e-01, 0.00000000e+00],
       [  2.87923038e-01, -2.87922770e-01, 0.00000000e+00],
       [  2.26219773e-01, -3.38561386e-01, 0.00000000e+00],
       [  1.55822948e-01, -3.76189291e-01, 0.00000000e+00],
       [  7.94379488e-02, -3.99360478e-01, 0.00000000e+00],
       [  1.76130357e-07, -4.07184392e-01, 0.00000000e+00],
       [ -7.94375911e-02, -3.99360538e-01, 0.00000000e+00],
       [ -1.55822605e-01, -3.76189470e-01, 0.00000000e+00],
       [ -2.26219490e-01, -3.38561565e-01, 0.00000000e+00],
       [  3.34985316e-01, 2.41472378e-07, 0.00000000e+00],
       [  1.28192902e-01, 3.09486121e-01, 0.00000000e+00],
       [  6.53519258e-02, 3.28548580e-01, 0.00000000e+00],
       [ -3.99360359e-01, -7.94377849e-02, 0.00000000e+00],
       [ -4.07184362e-01, -5.19040952e-08, 0.00000000e+00],
       [ -3.99360359e-01, 7.94376805e-02, 0.00000000e+00],
       [ -3.76189291e-01, 1.55822635e-01, 0.00000000e+00],
       [ -3.38561475e-01, 2.26219445e-01, 0.00000000e+00],
       [ -2.87922800e-01, 2.87922740e-01, 0.00000000e+00],
       [ -2.26219490e-01, 3.38561386e-01, 0.00000000e+00],
       [ -1.55822664e-01, 3.76189172e-01, 0.00000000e+00],
       [ -7.94376880e-02, 3.99360359e-01, 0.00000000e+00],
       [  4.34528431e-08, 4.07184333e-01, 0.00000000e+00],
       [  3.09486151e-01, -1.28193125e-01, 0.00000000e+00],
       [  2.78530240e-01, -1.86107725e-01, 0.00000000e+00],
       [  2.36870542e-01, -2.36870319e-01, 0.00000000e+00],
       [  1.86108038e-01, -2.78530061e-01, 0.00000000e+00],
       [  1.28193498e-01, -3.09486002e-01, 0.00000000e+00],
       [  6.53525665e-02, -3.28548640e-01, 0.00000000e+00],
       [  1.52604898e-07, -3.34985316e-01, 0.00000000e+00],
       [ -6.53522611e-02, -3.28548729e-01, 0.00000000e+00],
       [ -1.28193215e-01, -3.09486151e-01, 0.00000000e+00],
       [ -1.86107785e-01, -2.78530180e-01, 0.00000000e+00],
       [ -3.28548610e-01, -6.53524399e-02, 0.00000000e+00],
       [ -3.34985256e-01, -5.48066623e-08, 0.00000000e+00],
       [ -3.28548610e-01, 6.53523207e-02, 0.00000000e+00],
       [ -3.09485972e-01, 1.28193229e-01, 0.00000000e+00],
       [ -2.78530061e-01, 1.86107755e-01, 0.00000000e+00],
       [ -2.36870319e-01, 2.36870274e-01, 0.00000000e+00],
       [ -1.86107785e-01, 2.78530031e-01, 0.00000000e+00],
       [ -1.28193274e-01, 3.09485912e-01, 0.00000000e+00],
       [ -6.53523356e-02, 3.28548521e-01, 0.00000000e+00],
       [  4.34528431e-08, 3.34985197e-01, 0.00000000e+00],
       [  2.38739684e-01, 7.89269283e-02, 0.00000000e+00],
       [  4.65920269e-01, 1.54032513e-01, 0.00000000e+00],
       [ -4.65920389e-01, -1.54032156e-01, 0.00000000e+00],
       [ -2.38739669e-01, -7.89268017e-02, 0.00000000e+00],
       [  2.61837900e-01, 2.62953997e-01, 0.00000000e+00],
       [ -2.61838138e-01, -2.62953877e-01, 0.00000000e+00]]

    faces = [[8, 9, 32, 3], [29, 30, 50, 49], [24, 25, 45, 44], [15, 16, 39, 38],
    [10, 11, 34, 33], [26, 27, 47, 46], [17, 18, 41, 40], [12, 13, 36, 35], [30, 31, 51, 50],
    [28, 29, 49, 48], [14, 15, 38, 37], [0, 1, 20, 21], [31, 0, 21, 51], [16, 17, 40, 39],
    [7, 8, 3, 19], [25, 26, 46, 45], [23, 24, 44, 43], [9, 10, 33, 32], [27, 28, 48, 47],
    [11, 12, 35, 34], [6, 7, 19, 4], [1, 2, 5, 20], [22, 23, 43, 42], [13, 14, 37, 36],
    [22, 42, 55, 54], [6, 4, 52, 53], [54, 55, 57], [53, 52, 56]]

    rotate_2 = bpy.data.meshes.new("rotate_2")
    rotate_2.from_pydata(verts, [], faces)
    rotate_2.update()

    rotate_2_ob = bpy.data.objects.new("rotate_2", rotate_2)
    bpy.context.scene.objects.link(rotate_2_ob)
    rotate_2_ob.show_x_ray = True
    #    smooth = np.ones_like(rotate_2.polygons)
    #    rotate_2.polygons.foreach_set('use_smooth', smooth)
    tracker_2 = bpy.data.objects.new('tracker_2', None)
    bpy.context.scene.objects.link(tracker_2)
    tracker_2.scale = [0.001, 0.001, 0.001]
    tracker_2.location = rotate_2_ob.location    #    + normal
    constraint = rotate_2_ob.constraints.new('TRACK_TO')
    constraint.target = tracker_2
    constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'

    if 'rotate_2_color' not in bpy.data.materials:
        rotate_2_color = bpy.data.materials.new('rotate_2_color')
        rotate_2_color.diffuse_color = [1, 0, 0]
        rotate_2_color.use_transparency = True
        rotate_2_color.alpha = .673
        rotate_2_color.use_shadeless = True
    else:
        rotate_2_color = bpy.data.materials['rotate_2_color']

    rotate_2.materials.append(rotate_2_color)
