# Python built-in modules
from bisect import bisect_left
import random      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args


from core import Node
from transform import (lerp, quaternion, quaternion_from_euler, quaternion_slerp, quaternion_matrix, translate,
                       scale, identity, vec)


# -------------- Keyframing Utilities TP6 ------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time <= self.times[0]:
            return self.values[0]
        if time >= self.times[-1]:
            # return self.values[int(time % len(self.times))]
            return self.values[-1]
        # 2. search for closest index entry in self.times, using bisect_left
        t_i = bisect_left(self.times, time) - 1

        # 3. using the retrieved index, interpolate between the two neighboring
        # values in self.values, using the stored self.interpolate function
        f = (time - self.times[t_i]) / (self.times[t_i + 1] - self.times[t_i])
        return self.interpolate(self.values[t_i], self.values[t_i + 1], f)


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys, NodeName=None, boucle=10):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys)
        self.rotate_keys = KeyFrames(rotate_keys)
        self.scale_keys = KeyFrames(scale_keys)
        self.NodeName = NodeName
        self.boucle = boucle

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_mat = translate(self.translate_keys.value(time))
        rotate_mat = quaternion_matrix(self.rotate_keys.value(time % self.boucle))
        scale_mat = scale(self.scale_keys.value(time % self.boucle))
        if time > 41:
            glfw.set_time(0.0)
        return translate_mat @ rotate_mat @ scale_mat
        #return translate_mat @ scale_mat

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, trans_keys, rot_keys, scale_keys, name=None, transform=identity()):
        super().__init__(transform=transform)
        self.keyframes = TransformKeyFrames(trans_keys, rot_keys, scale_keys, name)
        self.name = name

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(primitives=primitives, **uniforms)

    def key_handler(self, key):
        lastPos = self.keyframes.translate_keys.value(glfw.get_time())
        list_key = (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_D, glfw.KEY_U)
        if type(self.name) == str and self.name == 'pointeur':
            if key in list_key :
                if key == glfw.KEY_DOWN:
                    transkey, rotkey, scalekey = sens_rotation(1, 0, 'pointeur', lastPos)   
                elif key == glfw.KEY_UP:
                    transkey, rotkey, scalekey = sens_rotation(-1, 0, 'pointeur', lastPos)     
                elif key == glfw.KEY_LEFT:
                    transkey, rotkey, scalekey = sens_rotation(-2, 0, 'pointeur', lastPos)     
                elif key == glfw.KEY_RIGHT:
                    transkey, rotkey, scalekey = sens_rotation(2, 0, 'pointeur', lastPos)
                elif key == glfw.KEY_D:
                    transkey, rotkey, scalekey = sens_rotation(-3, 0, 'pointeur', lastPos)     
                elif key == glfw.KEY_U:
                    transkey, rotkey, scalekey = sens_rotation(3, 0, 'pointeur', lastPos)      
                self.keyframes = TransformKeyFrames(transkey, rotkey, scalekey)
            
# -------------- Linear Blend Skinning : TP7 ---------------------------------
class Skinned:
    """ Skinned mesh decorator, passes bone world transforms to shader """
    def __init__(self, mesh, bone_nodes, bone_offsets):
        self.mesh = mesh

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = np.array(bone_offsets, np.float32)

    def draw(self, **uniforms):
        world_transforms = [node.world_transform for node in self.bone_nodes]
        uniforms['bone_matrix'] = world_transforms @ self.bone_offsets
        self.mesh.draw(**uniforms)

def sens_rotation(sens, angle, file=None, lastPos=[0, 0, 0], move=True):
    translate_keys = {}
    rotate_keys = {}
    scale_keys = {}
    if file == 'seagull':
        for i in range(100):
            translate_keys[i] = vec(0, 20 + i, 4.5 * i * sens)
            rotate_keys[i] = quaternion(0, 0, 0)
            scale_keys[i] = 1
        

    elif file == 'pointeur':
        if not move :
            for i in range(100):
                scale_keys[i] = 1
                rotate_keys[i] =quaternion()
                translate_keys[i] = vec(lastPos[0], lastPos[1], lastPos[2])
        else :
            for i in range(100):
                scale_keys[i] = 1
                if sens == -2 or sens == 2:
                    rotate_keys[i] = quaternion()
                    translate_keys[i] = vec(int(sens/2) + lastPos[0], lastPos[1], lastPos[2])
                elif abs(sens) == 1:
                    rotate_keys[i] = quaternion()
                    translate_keys[i] = vec(lastPos[0], lastPos[1], sens + lastPos[2])
                elif abs(sens) == 3:
                    rotate_keys[i] = quaternion()
                    translate_keys[i] = vec(lastPos[0], int(sens/3) + lastPos[1], lastPos[2])
    elif file == 'boat':
        for i in range(1000):
            translate_keys[i] = vec(10 * 2*np.cos(i), 0,  10 * 2*np.sin(i))
            if i == 0:
                rotate_keys[0] = quaternion()
            else :
                rotate_keys[i] = quaternion_from_euler(0, i * sens * angle, 0)
            scale_keys[i] = 1
    return translate_keys, rotate_keys, scale_keys