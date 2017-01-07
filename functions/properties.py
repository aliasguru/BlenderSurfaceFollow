'''
Created on 07.01.2017

@author: renderman
'''
import bpy



def run_handler(scene, override = False):
    multi_update()
    #    test_thingy()

def remove_handler(type):
    '''Deletes handler from the scene'''
    if type == 'scene':
        if run_handler in bpy.app.handlers.scene_update_pre:
            bpy.app.handlers.scene_update_pre.remove(run_handler)
    if type == 'frame':
        if run_handler in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(run_handler)

def add_handler(type):
    '''adds handler from the scene'''
    if type == 'scene':
        bpy.app.handlers.scene_update_pre.append(run_handler)
    if type == 'frame':
        bpy.app.handlers.frame_change_post.append(run_handler)


#    run on prop callback
def toggle_display(self, context):
    if bpy.context.scene.surface_follow_on:
        add_handler('scene')
        remove_handler('frame')
        bpy.context.scene['surface_follow_frame'] = False

    elif bpy.context.scene.surface_follow_frame:
        add_handler('frame')
        remove_handler('scene')
        bpy.context.scene['surface_follow_on'] = False
    else:
        remove_handler('scene')
        remove_handler('frame')


#    Properties-----------------------------------:
def create_properties():

    bpy.types.Scene.surface_follow_on = bpy.props.BoolProperty(name = "Scene Update",
        description = "For toggling the dynamic tension map",
        default = False, update = toggle_display)

    bpy.types.Scene.surface_follow_frame = bpy.props.BoolProperty(name = "Frame Update",
        description = "For toggling the dynamic tension map",
        default = False, update = toggle_display)

    bpy.types.Scene.surface_follow_data_set = {}
    bpy.types.Scene.surface_follow_data_set['surfaces'] = {}
    bpy.types.Scene.surface_follow_data_set['objects'] = {}

def remove_properties():
    '''Walks down the street and gets me a coffee'''
    del(bpy.types.Scene.surface_follow_on)
    del(bpy.types.Scene.surface_follow_frame)
    del(bpy.types.Scene.surface_follow_data_set)