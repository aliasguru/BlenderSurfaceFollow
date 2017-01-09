bl_info = {
    "name": "Texture Transformer",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 78, 0),
    "location": "View3D > Extended Tools > Adjust Active Texture",
    "description": "Lets you manipulate textures easily in the 3d view",
    "warning": "Experimental. Has been know to cause reasonable fear of clowns",
    "wiki_url": "",
    "category": '3D View'}

import bpy

import bmesh
from bpy_extras import view3d_utils
import mathutils
import numpy as np


def rotate_around_axis(coords, Q, origin = 'empty'):
    '''Uses standard quaternion to rotate a vector. Q requires
    a 4-dimensional vector. coords is the 3d location of the point.
    coords can also be an N x 3 array of vectors. Happens to work
    with Q as a tuple or a np array shape 4'''
    if origin == 'empty':
        vcV = np.cross(Q[1:], coords)
        RV = np.nan_to_num(coords + vcV * (2 * Q[0]) + np.cross(Q[1:], vcV) * 2)
    else:
        coords -= origin
        vcV = np.cross(Q[1:], coords)
        RV = (np.nan_to_num(coords + vcV * (2 * Q[0]) + np.cross(Q[1:], vcV) * 2)) + origin
        coords += origin    #    undo in-place offset
    return RV

def transform_matrix(V, ob = 'empty', back = False):
    '''Takes a vector and returns it with the
    object transforms applied. Also works
    on N x 3 array of vectors'''
    if ob == 'empty':
        ob = bpy.context.object
    ob.rotation_mode = 'QUATERNION'
    if back:
        rot = np.array(ob.rotation_quaternion)
        rot[1:] *= -1
        V -= np.array(ob.location)
        rotated = rotate_around_axis(V, rot)
        rotated /= np.array(ob.scale)
        return rotated

    rot = np.array(ob.rotation_quaternion)
    rotated = rotate_around_axis(V, rot)
    return np.array(ob.location) + rotated * np.array(ob.scale)

def get_active_uv_name(type = 'UV'):
    '''Finds the active uv layer and writes it's name to the dictionary'''
    ob = bpy.context.object
    mat_name = ob.active_material.name
    tex_idx = ob.data.materials[mat_name].active_texture_index
    tex_uv = bpy.data.materials[mat_name].texture_slots[tex_idx].uv_layer
    if type == 'tex_idx':
        return tex_idx

    if type == 'image':
        return bpy.data.materials[mat_name].texture_slots[tex_idx].texture.image
    if tex_uv == "":
        active = ob.data.uv_layers.active_index
        tex_uv = ob.data.uv_layers[active].name

    return tex_uv

def set_active_uv_name():
    '''Finds the active uv layer and writes it's name to the dictionary'''
    ob = bpy.context.object
    mat_name = ob.active_material.name
    tex_idx = ob.data.materials[mat_name].active_texture_index
    ttd = bpy.context.scene.texture_transform_data
    ttd['texture_index'] = tex_idx
    return bpy.data.materials[mat_name].texture_slots[tex_idx].uv_layer

def initialize_transform_data():
    sce = bpy.context.scene
    ob = bpy.context.object
    bpy.types.Scene.texture_transform_data = {}
    ttd = bpy.context.scene.texture_transform_data
    ttd['tracker_switch'] = True    #    for rotate gizmo to point at view
    ttd['static_uv'] = get_uv_coords(ob, get_active_uv_name())
    ttd['start_coords'] = np.copy(ttd['static_uv'])
    ttd['first_vec'] = ttd['start_coords'][[0, 1]]
    ttd['stored_uv_coords_full'] = np.copy(ttd['static_uv'])
    ttd['rot_offset'] = np.array([0.0, 0.0])
    ttd['scale_offset'] = np.array([0.0, 0.0])
    return

#    for i in ob.data.materials:
#        for j in i.texture_slots:
#            if j != None:
#                uv_name = get_active_uv_name()
#                image = j.texture.image
#                texture_name = j.texture.name

#                d['move_offset'] = np.array([0.0, 0.0])
#                d['stored_uv_coords_full'] = get_uv_coords(ob, uv_name)
#                d['start_coords'] = np.copy(d['stored_uv_coords_full'])
#                d['first_vec'] = d['start_coords'][[0,1]]
#                d['rotation_vectors'] = np.zeros(2*2*2).reshape(2,2,2)
#                d['image'] = image
#                d['uv_name'] = uv_name
#                d['image_size'] = image.size[1] / image.size[0]
#                d['pds_total_offset'] = np.array([0.0, 0.0])
#                d['pds_scale'] = np.array([1.0, 1.0])
#                d['pds_rotation_degrees'] = 0.0
#                if white_space:
#                    if texture_name in sce.image_centers.keys():
#                        dim = np.array(image.size, dtype=np.float64)
#                        to_center = np.array(sce.image_centers[texture_name]).astype(np.float64)
#                        # pds has the top left corner as 0.0, blender has the bottom left corner as 0.0
#                        to_center[1] = dim[1] - to_center[1]
#                        d['image_rotation_center'] = (to_center / dim)
#                    else:
#                        d['image_rotation_center'] = np.array([0.5, 0.5])
#                else:
#                    d['image_rotation_center'] = np.array([0.5, 0.5])



def get_quat_2(v1, v2, axis = 'shortest_arc', scalar = 1, rot_mag = 1):
    '''Measures the angle between two points and returns the quaternion
    that rotates the first point to the second. If an axis is given it gives
    the angle along that axis otherwise it uses shortest arc'''
    if axis == 'shortest_arc':
        Uv1 = v1 / np.sqrt(np.dot(v1, v1))
        Uv2 = v2 / np.sqrt(np.dot(v2, v2))
        mid = Uv1 + Uv2
        Umid = mid / np.sqrt(np.dot(mid, mid))
        w = np.dot(Umid, Uv1)
        xyz = np.cross(Umid, Uv1) * rot_mag
    else:
        vc1 = np.cross(axis, v1)
        vc2 = np.cross(axis, v2)
        Uv1 = vc1 / np.sqrt(np.dot(vc1, vc1))
        Uv2 = vc2 / np.sqrt(np.dot(vc2, vc2))
        mid = Uv1 + Uv2
        Umid = mid / np.sqrt(np.dot(mid, mid))
        w = np.dot(Umid, Uv1)
        xyz = np.cross(Umid, Uv2)
    if scalar != 1:
        theta = np.arccos(w)
        w = np.cos(theta * scalar)
        inv_xyz = xyz / np.sin(theta)
        xyz = inv_xyz * np.sin(theta * scalar)
    Q = np.append(w, -xyz)
    UQ = np.nan_to_num(Q / np.sqrt(np.dot(Q, Q)))
    return UQ

def rotate(Q, coords, origin = 'empty'):
    '''Uses standard quaternion to rotate a vector. Q requires
    a 4-dimensional vector. coords is the 3d location of the point.
    coords can also be an N x 3 array of vectors. Happens to work
    with Q as a tuple or a np array shape 4'''
    if origin == 'empty':
        vcV = np.cross(Q[1:], coords)
        RV = np.nan_to_num(coords + vcV * (2 * Q[0]) + np.cross(Q[1:], vcV) * 2)
    else:
        coords -= origin
        vcV = np.cross(Q[1:], coords)
        RV = (np.nan_to_num(coords + vcV * (2 * Q[0]) + np.cross(Q[1:], vcV) * 2)) + origin
    return RV

def get_uv_coords(ob, layer = None, proxy = False):
    '''Creates an N x 2 numpy array of uv coords. If proxy is used the
    coords are taken from the object specified with modifiers evaluated.
    For the proxy argument put in the object: get_coords(ob, proxy_ob)
    "layer='my_uv_map_name'" is the name of the uv layer you want to use.'''
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.uv_layers[layer].data
    else:
        verts = ob.data.uv_layers[layer].data
    v_count = len(verts)
    coords = np.zeros(v_count * 2, dtype = np.float64)
    verts.foreach_get("uv", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 2)

def set_coords(coords, ob, use_proxy = 'empty'):
    """Writes a flattened array to the object. Second argument is for reseting
    offsets created by modifiers to avoid compounding of modifier effects"""
    if use_proxy == 'empty':
        ob.data.vertices.foreach_set("co", coords.ravel())
    else:
        coords += use_proxy
        ob.data.vertices.foreach_set("co", coords.ravel())
    ob.data.update()

def get_coords(ob, proxy = False):
    '''Creates an N x 3 numpy array of vertex coords. If proxy is used the
    coords are taken from the object specified with modifiers evaluated.
    For the proxy argument put in the object: get_coords(ob, proxy_ob)'''
    if proxy:
        mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.vertices
    else:
        verts = ob.data.vertices
    v_count = len(verts)
    coords = np.zeros(v_count * 3, dtype = np.float64)
    verts.foreach_get("co", coords)
    if proxy:
        bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 3)

def reset_rotation():
    pass
    #    Would need to create an item for each uv map and check it for changes and stuff. Lots of work.
    #    ttd = bpy.context.scene.texture_transform_data
    #    ttd['rotation_vectors'] = np.zeros(8).reshape(2,2,2)
    #    set_uv_coords(ttd['start_coords'], ttd['static_uv_name'], ob)
    #    ttd['stored_uv_coords_full'] = ttd['start_coords']
    #    offset = bpy.context.object.active_material.texture_slots[ttd['texture_index']].offset
    #    offset.xy = np.array(offset.xy) - ttd['rot_offset']
    #    ttd['rot_offset'] = np.array([0.0, 0.0])

def reset_move():
    set_active_uv_name()
    ttd = bpy.context.scene.texture_transform_data
    offset = bpy.context.object.active_material.texture_slots[ttd['texture_index']].offset
    offset.xy = np.array(offset.xy) - ttd['move_offset']
    ttd['move_offset'] = np.array([0.0, 0.0])

def reset_scale():
    set_active_uv_name()
    ttd = bpy.context.scene.texture_transform_data
    offset = bpy.context.object.active_material.texture_slots[ttd['texture_index']].offset
    offset.xy = np.array(np.array([0.0, 0.0]))
    ttd['scale_offset'] = np.array([0.0, 0.0])
    scale = bpy.context.object.active_material.texture_slots[ttd['texture_index']].scale
    ttd['pds_total_offset'] = np.array(offset.xy)
    ttd['pds_scale'] = np.array([1.0, 1.0])
    scale.xy = np.array([1.0, 1.0])

def reset_all():
    reset_rotation()
    reset_scale()
    reset_move()

def set_uv_coords(coords, layer, ob):
    """Writes a flattened array to the object."""
    ob.data.uv_layers[layer].data.foreach_set("uv", coords.ravel())
    ob.data.update()

def raycast(hit, success = False, uv = None, normal = None):
    ttd = bpy.context.scene.texture_transform_data
    if success:
        uv_hit = np.array(uv)[0:2]
        mesh = bpy.context.object
        if ttd['stored_uv_hit'] == 'empty':
            ttd['stored_uv_hit'] = np.copy(uv_hit)

        dif = uv_hit - ttd['stored_uv_hit']
        scale = mesh.active_material.texture_slots[ttd['texture_index']].scale
        offset = mesh.active_material.texture_slots[ttd['texture_index']].offset
        image_compensate = np.array([1, ttd['image_size']])    #    * np.array(scale.xy)# for difference in image xy scale (non-square)
        compensate_inverse = np.array([1, 1 / ttd['image_size']])    #    * (1/np.array(scale.xy))
        current_loc = np.array(offset.xy)
        # -----------------------------------------------------------------
        if ttd['type'] == 'move':
            if (np.fabs(dif[0]) > .001) or (np.fabs(dif[1]) > .001):
                if ttd['move_axis'] == 'empty':
                    ttd['move_axis'] = np.argmin(np.fabs(dif))
                if ttd['con_xy']:
                    dif[ttd['move_axis']] *= 0
            saved_move = np.copy(offset.xy)
            offset.xy = current_loc - dif * np.array(scale.xy)    #    * compensate_inverse
            move_dif = np.array(offset.xy) - saved_move
            ttd['move_offset'] += move_dif
            ttd['stored_uv_hit'] = np.copy(uv_hit)
        # -----------------------------------------------------------------
        elif ttd['type'] == 'scale':
            saved_scale_offset = np.copy(offset.xy)    #    store the offset seperately for the reset tool
            factor = bpy.context.scene.scale_strength
            if ttd['h1'] == 'empty':
                ttd['h1'] = np.copy(uv_hit) * image_compensate
                ttd['current_scale'] = np.array(scale.xy)
                ttd['current_loc'] = np.array(offset.xy)
            if ttd['h1'] != 'empty':
                if ttd['h2'] == 'empty':
                    dif = ttd['h1'] * compensate_inverse - uv_hit
                    mid = np.array([0.5, 0.5]) * image_compensate
                    to_mid = (mid - ttd['h1']) * np.array(scale.xy)
                    offset.xy = ttd['current_loc'] - to_mid * compensate_inverse
                    if ttd['scale_xy']:
                        abs_x = np.fabs(dif[0])
                        abs_y = np.fabs(dif[1])
                        if ttd['scale_axis'] == 'empty':
                            if (abs_x > .001) or (abs_y > .001):
                                ttd['scale_axis'] = np.argmin([abs_x, abs_y])
                        if ttd['scale_axis'] != 'empty':
                            dif[ttd['scale_axis']] *= 0
                            scale.xy = ttd['current_scale'] + dif * ttd['current_scale'] * factor
                    elif ttd['scale_free']:
                        scale.xy = ttd['current_scale'] + dif * ttd['current_scale'] * factor
                    else:
                        xy_dif = scale.x / scale.y    #    still scales evenly when x and y are different
                        dif = (ttd['mouse_y_start'] - ttd['mouse_y']) * .0005
                        mean = np.mean(dif * ttd['current_scale'])
                        mix = np.array([mean * xy_dif, mean])
                        scale.xy = ttd['current_scale'] + mix * factor

                    to_mid = (mid - ttd['h1']) * np.array(scale.xy)
                    offset.xy = np.array(offset.xy) + to_mid * compensate_inverse
                    ttd['current_loc'] = np.array(offset.xy)

        # -------------------------------------------------------------------

        elif ttd['type'] == 'rotate':
            gizmo = bpy.data.objects['rotate_1']
            gizmo_coords = get_coords(gizmo)
            if ttd['h1'] == 'empty':
                bpy.data.objects['rotate_1'].hide = False
                ttd['h1'] = np.copy(uv_hit)
                ttd['start_mid'] = np.array([0.5, 0.5])
                ttd['start_current'] = np.copy(offset.xy) * (1 / np.array(scale.xy))
                bpy.data.objects['rotate_1'].location = transform_matrix(hit)    #    np.array(bpy.context.object.matrix_world.inverted()).T[3, :3]
            if ttd['h1'] != 'empty':
                if ttd['h2'] == 'empty':
                    dif = ttd['h1'] - uv_hit
                    if np.sqrt(np.dot(dif, dif)) > 0.01:
                        ttd['h2'] = np.copy(uv_hit)
                        if 'rotate_2' not in bpy.data.objects:
                            create_rotate_2()
                        bpy.data.objects['rotate_2'].hide = False
                if ttd['h2'] != 'empty':
                    v1 = (ttd['h2'] - ttd['h1']) * image_compensate
                    v2 = (uv_hit - ttd['h1']) * image_compensate
                    if ttd['increment']:
                        Uv1 = v1 / np.sqrt(np.dot(v1, v1))
                        Uv2 = v2 / np.sqrt(np.dot(v2, v2))
                        angle = np.arccos(np.dot(Uv1, Uv2))
                        if angle > np.pi / 9.7:
                            #------------------------------
                            sign = np.sign(np.cross(np.append(v2, 0), np.append(v1, 0))[2])
                            Q = np.array([0.98078528040323043, 0.0, 0.0, 0.19509032 * sign])    #    22.5 degree quat
                            mid = np.append(np.array([0.5, 0.5]) * image_compensate, 1)
                            #-------------------------
                            with_z = np.insert(ttd['stored_uv_coords_full'], 2, 0, axis = 1)
                            #    Rotate Gizmo:
                            rot_gizmo = rotate(Q * np.array([-1, 1, 1, 1]), gizmo_coords)
                            #    rot_gizmo = rotate(Q, gizmo_coords)
                            set_coords(rot_gizmo, gizmo)
                            #------------------------------
                            rot_coords = rotate(Q, with_z * np.append(image_compensate, 1),
                                origin = mid) * np.append(compensate_inverse, 1)

                            ori = ttd['h1'] * image_compensate
                            rot_mid = rotate(Q, np.append(ttd['start_mid'] * image_compensate, 0),
                                origin = np.append(ori, 0))[:2] * compensate_inverse

                            #    store rotation offset seperately for reset_rotation()
                            ttd['start_mid'] = rot_mid
                            saved = np.copy(offset.xy)
                            offset.xy = (rot_mid - np.array([0.5, 0.5]) + ttd['start_current']) * np.array(scale.xy)
                            dif = np.array(offset.xy) - saved
                            ttd['rot_offset'] += dif

                            #    ** Apply transforms in viewport
                            ttd['h2'] = np.copy(uv_hit)
                            ttd['stored_uv_coords_full'] = np.copy(rot_coords[:, :2])
                            set_uv_coords(rot_coords[:, :2], ttd['static_uv_name'], ob = bpy.context.object)

                    else:
                        mid = np.append(np.array([0.5, 0.5]) * image_compensate, 1)
                        #-------------------------
                        Q = get_quat_2(np.append(v2, 0), np.append(v1, 0), rot_mag = ttd['rot_scale'])
                        with_z = np.insert(ttd['stored_uv_coords_full'], 2, 0, axis = 1)
                        #    Rotate Gizmo:
                        rot_gizmo = rotate(Q * np.array([-1, 1, 1, 1]), gizmo_coords)
                        #    rot_gizmo = rotate(Q, gizmo_coords)
                        set_coords(rot_gizmo, gizmo)
                        #------------------------------
                        rot_coords = rotate(Q, with_z * np.append(image_compensate, 1),
                            origin = mid) * np.append(compensate_inverse, 1)

                        ori = ttd['h1'] * image_compensate
                        rot_mid = rotate(Q, np.append(ttd['start_mid'] * image_compensate, 0),
                            origin = np.append(ori, 0))[:2] * compensate_inverse

                        #    store rotation offset seperately for reset_rotation()
                        ttd['start_mid'] = rot_mid
                        saved_rot = np.copy(offset.xy)
                        offset.xy = (rot_mid - np.array([0.5, 0.5]) + ttd['start_current']) * np.array(scale.xy)
                        dif = np.array(offset.xy) - saved_rot
                        ttd['rot_offset'] += dif

                        #    ** Apply transforms in viewport
                        ttd['h2'] = np.copy(uv_hit)
                        ttd['stored_uv_coords_full'] = np.copy(rot_coords[:, :2])
                        set_uv_coords(rot_coords[:, :2], ttd['static_uv_name'], ob = bpy.context.object)
     # ------------------------------------------------
       # ------------------------------------------------

def drag_textures():
    ob = bpy.context.object
    set_active_uv_name()    #    sets ttd['uv_name'] to the active uv map
    ttd = bpy.context.scene.texture_transform_data
    ttd['static_uv_name'] = get_active_uv_name()
    ttd['uv_offseter'] = np.array([len(i.vertices) for i in ob.data.polygons])
    ttd['stored_uv_hit'] = 'empty'
    ttd['constrain'] = False    #    for checking the transform constraint
    ttd['even'] = True    #    for checking the transform constraint
    ttd['rot_scale'] = 1    #    for rotating slowly
    ttd['increment'] = False    #    for rotating by 22.5 degree increments
    ttd['con_xy'] = False
    ttd['scale_xy'] = False
    ttd['scale_free'] = False
    ttd['move_axis'] = 'empty'
    ttd['static_uv'] = get_uv_coords(ob, ttd['static_uv_name'])
    image = get_active_uv_name('image')
    ttd['image_size'] = image.size[1] / image.size[0]
    ttd['move_offset'] = np.array([0.0, 0.0])
    ttd['rot_offset'] = np.array([0.0, 0.0])


def main(drag, context, event):
    """Run this function on left mouse, execute the ray cast"""
    ttd = bpy.context.scene.texture_transform_data
    obj = bpy.context.object
    scene = context.scene
    scene.objects.active = obj
    mode = bpy.context.object.mode
    if mode != 'OBJECT':
        bpy.ops.object.mode_set(mode = 'OBJECT')

    #    get the ray from the viewport and mouse
    region = context.region
    rv3d = context.region_data
    coord = event.mouse_region_x, event.mouse_region_y
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    clip = bpy.context.space_data.clip_end
    val = .05
    if clip < 150:
        bpy.context.space_data.clip_end = 150
    if clip > 1000:
        bpy.context.space_data.clip_end = 1000
    if rv3d.view_perspective == 'ORTHO':
        ray_start = origin + view_vector * -100    #    ray starts 100 units from zero vector
        ray_end = origin + view_vector * 1000    #    ray ends 100 units past zero vector
        if ttd['type'] == 'move':
            bpy.data.objects['arrows'].location = origin + view_vector * 3
            bpy.data.objects['tracker'].location = origin + view_vector * -3
            bpy.data.objects['arrows'].scale = np.array([val, val, val])

        if ttd['type'] == 'scale':
            bpy.data.objects['scale'].location = origin + (view_vector * 50)
            bpy.data.objects['tracker'].location = origin + (view_vector * -5)
            bpy.data.objects['scale'].scale = np.array([val, val, val])

        if ttd['type'] == 'rotate':
            bpy.data.objects['rotate_0'].location = origin + (view_vector * 50)
            bpy.data.objects['tracker_0'].location = origin + (view_vector * -5)
            bpy.data.objects['rotate_0'].scale = np.array([val, val, val])
            if 'tracker_1' in bpy.data.objects:
                bpy.data.objects['tracker_1'].location = origin + (view_vector * +5)
                bpy.data.objects['rotate_1'].rotation_mode = 'QUATERNION'
                if ttd['tracker_switch'] == True:
                    ttd['tracker_switch'] = False
                    ttd['view_vector'] = view_vector    #    send to racaster to rotate gizmo with existing quaternion

            if 'rotate_2' in bpy.data.objects:
                bpy.data.objects['rotate_2'].location = origin + (view_vector * 50)
                bpy.data.objects['tracker_2'].location = origin + (view_vector * -5)
                bpy.data.objects['rotate_2'].scale = np.array([val, val, val])

    else:
        ray_start = origin + view_vector - view_vector * 2
        ray_end = ray_start + (view_vector) * 120
        if ttd['type'] == 'move':
            bpy.data.objects['arrows'].location = origin + (view_vector * 50)
            bpy.data.objects['tracker'].location = origin + (view_vector * -5)
        if ttd['type'] == 'scale':
            bpy.data.objects['scale'].location = origin + (view_vector * 50)
            bpy.data.objects['tracker'].location = origin + (view_vector * -5)
        if ttd['type'] == 'rotate':
            bpy.data.objects['rotate_0'].location = origin + (view_vector * 50)
            bpy.data.objects['tracker_0'].location = origin + (view_vector * -5)
            if 'tracker_1' in bpy.data.objects:
                bpy.data.objects['tracker_1'].location = origin + (view_vector * -5)
                bpy.data.objects['rotate_1'].rotation_mode = 'QUATERNION'
                if ttd['tracker_switch'] == True:
                    ttd['tracker_switch'] = False
                    ttd['view_vector'] = view_vector    #    send to racaster to rotate gizmo with existing quaternion
            if 'rotate_2' in bpy.data.objects:
                bpy.data.objects['rotate_2'].location = origin + (view_vector * 50)
                bpy.data.objects['tracker_2'].location = origin + (view_vector * -5)

    def obj_ray_cast(obj, matrix):
        """Wrapper for ray casting that moves the ray into object space"""
        #    get the ray relative to the object
        matrix_inv = matrix.inverted()    #    this applies the opposite of the object transforms
        ray_start_obj = matrix_inv * ray_start
        ray_end_obj = matrix_inv * ray_end
        view_vec = ray_end_obj - ray_start_obj
        success, location, normal, face_index = obj.ray_cast(ray_start_obj, (view_vec) * 50000)

        v1, v2, v3 = (obj.data.polygons[face_index].vertices[i] for i in range(3))

        offset = np.sum(ttd['uv_offseter'][: face_index])
        layer = ttd['static_uv_name']

        coords = get_coords(obj)
        uv_coords = ttd['static_uv']

        obj.data.uv_layers[layer].data.update()
        uv1, uv2, uv3 = (np.append(uv_coords[offset + i], 0) for i in [0, 1, 2])

        bary = mathutils.geometry.barycentric_transform(location, coords[v1], coords[v2], coords[v3], uv1, uv2, uv3)
        raycast(np.array(location), success, bary, normal)

    if drag:
        matrix = obj.matrix_world.copy()    #    this is a 4x4 matrix storing the location rotation(as 3x3 matrix) and scale of the object
        obj_ray_cast(obj, matrix)

##################################################################################
class ResetRotation(bpy.types.Operator):
    '''Resets rotation and offset created by the rotate tool'''
    bl_idname = "object.reset_rotation"
    bl_label = "reset_rotation"
    def execute(self, context):
        reset_rotation()
        return {'FINISHED'}

class ResetMove(bpy.types.Operator):
    '''Resets offset created by the move tool'''
    bl_idname = "object.reset_move"
    bl_label = "reset_move"
    def execute(self, context):
        reset_move()
        return {'FINISHED'}

class ResetScale(bpy.types.Operator):
    '''Resets offset and scale created by the scale tool'''
    bl_idname = "object.reset_scale"
    bl_label = "reset_scale"
    def execute(self, context):
        reset_scale()
        return {'FINISHED'}

class ResetAll(bpy.types.Operator):
    '''Resets all adjustments'''
    bl_idname = "object.reset_all"
    bl_label = "reset_all"
    def execute(self, context):
        reset_all()
        return {'FINISHED'}

class RotateTextureModal(bpy.types.Operator):
    """Modal texture rotating tool"""
    bl_idname = "view3d.texture_rotate_modal"
    bl_label = "Rotate Texture"
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        self.rotate = False
        drag_textures()
        create_rotate_0()
        ttd = bpy.context.scene.texture_transform_data
        ttd['type'] = 'rotate'

    def modal(self, context, event):
        ttd = bpy.context.scene.texture_transform_data
        bpy.context.scene.tex_rotate_alert = True
        ttd['tracker_switch'] = False
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
        'NUMPAD_PERIOD', 'NUMPAD_2', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_8',
         'NUMPAD_1', 'NUMPAD_3', 'NUMPAD_5', 'NUMPAD_7', 'NUMPAD_9'}:
            if event.value == 'PRESS':
                bpy.data.objects['rotate_0'].hide = True
                if 'rotate_1' in bpy.data.objects:
                    bpy.data.objects['rotate_1'].hide = True
            if event.value == 'RELEASE':
                bpy.data.objects['rotate_0'].hide = False
                if 'rotate_1' in bpy.data.objects:
                    bpy.data.objects['rotate_1'].hide = False
            #    allow navigation
            return {'PASS_THROUGH'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if 'rotate_1' not in bpy.data.objects:
                create_rotate_1()

            ttd = bpy.context.scene.texture_transform_data
            ttd['static_uv'] = get_uv_coords(bpy.context.object, ttd['static_uv_name'])
            ttd['stored_uv_coords_full'] = get_uv_coords(bpy.context.object, ttd['static_uv_name'])
            ttd['type'] = 'rotate'
            ttd['h1'] = 'empty'
            ttd['h2'] = 'empty'
            bpy.data.objects['rotate_0'].hide = True
            self.rotate = True
            ttd['tracker_switch'] = True

        if event.type == 'LEFT_SHIFT' and event.value == 'PRESS':
            ttd['rot_scale'] = 0.1

        if event.type == 'LEFT_SHIFT' and event.value == 'RELEASE':
            ttd['rot_scale'] = 1

        if event.type == 'LEFT_CTRL' and event.value == 'PRESS':
            ttd['increment'] = True

        if event.type == 'LEFT_CTRL' and event.value == 'RELEASE':
            ttd['increment'] = False

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if 'rotate_1' in bpy.data.objects:
                bpy.data.objects['rotate_1'].hide = True
            if 'rotate_2' in bpy.data.objects:
                bpy.data.objects['rotate_2'].hide = True
            self.rotate = False

        elif event.type == 'MOUSEMOVE':
            main(self.rotate, context, event)
            if not self.rotate:
                bpy.data.objects['rotate_0'].hide = False

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            bpy.context.scene.tex_rotate_alert = False
            for i in bpy.data.objects:
                if i.name.startswith('rotate') or i.name.startswith('tracker'):
                    i.hide = False
                i.select = i.name.startswith('rotate') or i.name.startswith('tracker')
            bpy.ops.object.delete()
            meshes = [m for m in bpy.data.meshes if m.name.startswith('rotate')]
            for i in meshes:
                bpy.data.meshes.remove(i)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

###########################################################################

class ScaleTextureModal(bpy.types.Operator):
    """Modal texture scaling tool"""
    bl_idname = "view3d.texture_scale_modal"
    bl_label = "Scale Texture"
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        self.scale = False
        drag_textures()
        create_scale()
        ttd = bpy.context.scene.texture_transform_data
        ttd['type'] = 'scale'

    def modal(self, context, event):
        ttd = bpy.context.scene.texture_transform_data
        bpy.context.scene.tex_scale_alert = True
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
        'NUMPAD_PERIOD', 'NUMPAD_2', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_8',
         'NUMPAD_1', 'NUMPAD_3', 'NUMPAD_5', 'NUMPAD_7', 'NUMPAD_9'}:
            bpy.data.objects['scale'].hide = True
            #    allow navigation
            return {'PASS_THROUGH'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            bpy.data.objects['scale'].hide = True
            self.scale = True
            ttd['mouse_y_start'] = event.mouse_region_y
            ttd['scale_axis'] = 'empty'
            ttd['h1'] = 'empty'
            ttd['h2'] = 'empty'

        if event.type == 'LEFT_SHIFT' and event.value == 'PRESS':
            ttd['scale_xy'] = True
            ttd['scale_axis'] = 'empty'
            ttd['h1'] = 'empty'
            ttd['h2'] = 'empty'
        if event.type == 'LEFT_SHIFT' and event.value == 'RELEASE':
            ttd['scale_xy'] = False
            ttd['scale_axis'] = 'empty'
            ttd['h1'] = 'empty'
            ttd['h2'] = 'empty'

        if event.type == 'LEFT_CTRL' and event.value == 'PRESS':
            ttd['scale_free'] = True
            ttd['scale_axis'] = 'empty'
            ttd['h1'] = 'empty'
            ttd['h2'] = 'empty'

        if event.type == 'LEFT_CTRL' and event.value == 'RELEASE':
            ttd['scale_free'] = False
            ttd['scale_axis'] = 'empty'
            ttd['h1'] = 'empty'
            ttd['h2'] = 'empty'

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.scale = False
            bpy.data.objects['scale'].hide = False
            ttd['axis'] = 'empty'

        elif event.type == 'MOUSEMOVE':
            ttd['mouse_y'] = event.mouse_region_y
            main(self.scale, context, event)
            if not self.scale:
                bpy.data.objects['scale'].hide = False
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            bpy.context.scene.tex_scale_alert = False
            for i in bpy.data.objects:
                i.select = i.name.startswith('scale') or i.name.startswith('tracker')
            bpy.ops.object.delete()
            meshes = [m for m in bpy.data.meshes if m.name.startswith('scale')]
            for i in meshes:
                bpy.data.meshes.remove(i)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class MoveTextureModal(bpy.types.Operator):    #    a modal operator needs a function called 'modal' and one called 'invoke'
    """Modal texture moving tool"""
    bl_idname = "view3d.texture_move_modal"
    bl_label = "Move Texture"
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        self.move = False
        drag_textures()
        create_arrows()
        ttd = bpy.context.scene.texture_transform_data
        ttd['type'] = 'move'
        bpy.context.scene.tex_move_alert = True

    def modal(self, context, event):
        ttd = bpy.context.scene.texture_transform_data
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
        'NUMPAD_PERIOD', 'NUMPAD_2', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_8',
         'NUMPAD_1', 'NUMPAD_3', 'NUMPAD_5', 'NUMPAD_7', 'NUMPAD_9'}:
            bpy.data.objects['arrows'].hide = True
            #    allow navigation
            return {'PASS_THROUGH'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            ttd['stored_uv_hit'] = 'empty'
            bpy.data.objects['arrows'].hide = True
            self.move = True

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.move = False
            bpy.data.objects['arrows'].hide = False

        if event.type == 'LEFT_SHIFT' and event.value == 'PRESS':
            ttd['con_xy'] = True

        if event.type == 'LEFT_SHIFT' and event.value == 'RELEASE':
            ttd['con_xy'] = False
            ttd['move_axis'] = 'empty'

        elif event.type == 'MOUSEMOVE':
            main(self.move, context, event)
            if not self.move:
                bpy.data.objects['arrows'].hide = False
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            bpy.context.scene.tex_move_alert = False
            for i in bpy.data.objects:
                i.select = i.name.startswith('arrows') or i.name.startswith('tracker')
            bpy.ops.object.delete()
            meshes = [m for m in bpy.data.meshes if m.name.startswith('arrows')]
            for i in meshes:
                bpy.data.meshes.remove(i)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class MoveTexturePanel(bpy.types.Panel):
    """Creates Buttons for Line Tools"""
    bl_label = "Adjust Active Texture"
    bl_idname = "drag_textures"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Extended Tools"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align = True)
        #    col.operator("object.reset_all", text="Reset All Adjustments", icon='RECOVER_LAST')
        col = layout.column(align = True)

        move_text = "Move Texture"
        if bpy.context.scene.tex_move_alert:
            move_text = 'Right click to exit'
            col.alert = True

        col.operator("view3d.texture_move_modal", text = move_text, icon = 'HAND')
        col.operator("object.reset_move", text = "Reset Move", icon = 'RECOVER_LAST')
        col.label(text = "Left Shift for X Y")
        col = layout.column(align = True)

        scale_text = "Scale Texture"
        if bpy.context.scene.tex_scale_alert:
            scale_text = 'Right click to exit'
            col.alert = True
        #    col.label(text="Right Click To Exit")
        col.operator("view3d.texture_scale_modal", text = scale_text, icon = 'MOD_ARRAY')
        col.prop(bpy.context.scene , "scale_strength", text = "Scale Strength", slider = True)
        col.operator("object.reset_scale", text = "Reset Scale", icon = 'RECOVER_LAST')
        col.label(text = "Left Shift for X Y")
        col.label(text = "Left Ctrl for Free")
        #    col.label(text="Right Click To Exit")
        col = layout.column(align = True)

        rotate_text = "Rotate Texture"
        if bpy.context.scene.tex_rotate_alert:
            rotate_text = 'Right click to exit'
            col.alert = True
        col.operator("view3d.texture_rotate_modal", text = rotate_text, icon = 'FILE_REFRESH')
        #    col.operator("object.reset_rotation", text="Reset Rotation", icon='RECOVER_LAST')
        col.label(text = "Left Shift For Precise")
        col.label(text = "Left Ctrl for increment")
        #    col.label(text="Right Click To Exit")

def create_properties():
    bpy.types.Scene.scale_strength = bpy.props.FloatProperty(name = "Scale Strength",
        description = "Multiplies scale amount.", default = 10, min = 0.1, max = 20, precision = 2)
    bpy.types.Scene.texture_transform_data = {}

    bpy.types.Scene.tex_move_alert = bpy.props.BoolProperty(name = 'move alert', default = False)
    bpy.types.Scene.tex_scale_alert = bpy.props.BoolProperty(name = 'scale alert', default = False)
    bpy.types.Scene.tex_rotate_alert = bpy.props.BoolProperty(name = 'rotate alert', default = False)

def remove_properties():
    """I once traded a hamster for two-and-a-half all-season radial tires"""
    del(bpy.types.Scene.scale_strength)
    del(bpy.types.Scene.texture_transform_data)
    del(bpy.types.Scene.tex_move_alert)
    del(bpy.types.Scene.tex_scale_alert)
    del(bpy.types.Scene.tex_rotate_alert)

def register():
    create_properties()
    bpy.utils.register_class(MoveTexturePanel)
    bpy.utils.register_class(MoveTextureModal)
    bpy.utils.register_class(RotateTextureModal)
    bpy.utils.register_class(ScaleTextureModal)
    #    bpy.utils.register_class(ResetRotation)
    bpy.utils.register_class(ResetMove)
    bpy.utils.register_class(ResetScale)
    #    bpy.utils.register_class(ResetAll)

def unregister():
    remove_properties()
    bpy.utils.unregister_class(MoveTexturePanel)
    bpy.utils.unregister_class(MoveTextureModal)
    bpy.utils.unregister_class(RotateTextureModal)
    bpy.utils.unregister_class(ScaleTextureModal)
    #    bpy.utils.unregister_class(ResetRotation)
    bpy.utils.unregister_class(ResetMove)
    bpy.utils.unregister_class(ResetScale)
    #    bpy.utils.unregister_class(ResetAll)


if __name__ == "__main__":
    register()
