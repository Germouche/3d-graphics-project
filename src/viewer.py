#!/usr/bin/env python3

from re import S
import sys
from itertools import cycle
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np
import scipy as sp                  # all matrix manipulations & OpenGL args
from core import Node, Shader, Viewer, Mesh, load, Mannequin
from animation import KeyFrameControlNode, Skinned, sens_rotation
from texture import Texture, Textured
from transform import identity, rotate, vec, sincos, quaternion, quaternion_from_euler, scale, translate
from math import cos, sin

# For the drawing of the sphere
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")

M_PI =  3.14159265358979323846

# -------------- Example textured plane class ---------------------------------

class TexturedPlane(Textured):
    """ Simple first textured object """
    def __init__(self, shader, tex_file1, base_coords, tex_file2=None):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file1 = tex_file1
        self.file2 = tex_file2

        # setup plane mesh to be textured
        self.base_coords = base_coords
        #scaled = 100 * np.array(base_coords, np.float32)
        indices = np.array((0, 1, 2, 0, 2, 3), np.uint32)
        mesh = Mesh(shader, attributes=dict(position=base_coords), index=indices)

        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        texture1 = Texture(tex_file1, self.wrap, *self.filter)
        if (tex_file2 is not None):
            texture2 = Texture(tex_file2, self.wrap, *self.filter)
            super().__init__(mesh, diffuse_map=texture1, second_texture=texture2)
        else:
            super().__init__(mesh, diffuse_map=texture1)
            
    def key_handler(self, key):
        # cycle through texture modes on keypress of F6 (wrap) or F7 (filtering)
        self.wrap = next(self.wraps) if key == glfw.KEY_F6 else self.wrap
        self.filter = next(self.filters) if key == glfw.KEY_F7 else self.filter
        if key in (glfw.KEY_F6, glfw.KEY_F7):
            texture1 = Texture(self.file1, self.wrap, *self.filter)
            if (self.file2 is not None):
                texture2 = Texture(self.file2, self.wrap, *self.filter)
                self.textures.update(diffuse_map=texture1, second_texture=texture2)
            else:
                self.textures.update(diffuse_map=texture1)


# ------------------ Water class ----------------

class Water(Node):
    """ Water class """
    def __init__(self, children=(), transform=identity()):
        super().__init__(children, transform)

    def draw(self, model=identity(), **other_uniforms):
        """ Recursive draw, passing down updated model matrix. """
        other_uniforms['constante'] = 0.01 * glfw.get_time()
        super().draw(model, **other_uniforms)



# -------------- Deformable Cylinder Mesh  ------------------------------------


class SkinnedCylinder(KeyFrameControlNode):
    """ Deformable cylinder """
    def __init__(self, shader, sections=11, quarters=20):

        # this "arm" node and its transform serves as control node for bone 0
        # we give it the default identity keyframe transform, doesn't move
        super().__init__({0: (0, 0, 0)}, {0: quaternion()}, {0: 1})

        # we add a son "forearm" node with animated rotation for the second
        # part of the cylinder
        self.add(KeyFrameControlNode(
            {0: (0, 0, 0)},
            {0: quaternion(), 1: quaternion_from_euler(60), 2: quaternion()},
            {0: 1}))

        # there are two bones in this animation corresponding to above noes
        bone_nodes = [self, self.children[0]]

        # these bones have no particular offset transform
        bone_offsets = [identity(), identity()]

        # vertices, per vertex bone_ids and weights
        vertices, faces, bone_id, bone_weights = [], [], [], []
        for x_c in range(sections+1):
            for angle in range(quarters):
                z_c, y_c = sincos(360 * angle / quarters)
                vertices.append((x_c - sections/2, y_c, z_c))

                bone_id.append((0, 1, 0, 0))


                weight = 1 - x_c/sections
                #weight = 1 if x_c <= sections/2 else 0
                bone_weights.append((weight, 1 - weight, 0, 0))

        # face indices
        faces = []
        for x_c in range(sections):
            for angle in range(quarters):

                # indices of the 4 vertices of the current quad, % helps
                # wrapping to finish the circle sections
                ir0c0 = x_c * quarters + angle
                ir1c0 = (x_c + 1) * quarters + angle
                ir0c1 = x_c * quarters + (angle + 1) % quarters
                ir1c1 = (x_c + 1) * quarters + (angle + 1) % quarters

                # add the 2 corresponding triangles per quad on the cylinder
                faces.extend([(ir0c0, ir0c1, ir1c1), (ir0c0, ir1c1, ir1c0)])

        # the skinned mesh itself. it doesn't matter where in the hierarchy
        # this is added as long as it has the proper bone_node table
        attributes = dict(position=vertices, normal=bone_weights,
                          bone_ids=bone_id, bone_weights=bone_weights)
        mesh = Mesh(shader, attributes=attributes, index=faces)
        self.add(Skinned(mesh, bone_nodes, bone_offsets))


# -------------- Sphere2 class ---------------------------------

class Sphere2(Mesh):
    def __init__(self, shader, n_slices,  n_stacks, rayon, x_translation, y_translation, z_translation):

        self.index = []
        self.position = [(x_translation, 1*rayon + y_translation, z_translation), (x_translation, -1*rayon + y_translation, z_translation)]

        # add top vertex

        # generate vertices per stack / slice
        for i in range(0, n_stacks-1):
            phi = M_PI * (i+1) / n_stacks
            for j in range(0, n_slices):
                theta = 2.0 * M_PI * j / n_slices;
                x = rayon*(sin(phi) * cos(theta)) + x_translation #on multiplie par le rayon
                y = rayon*cos(phi) + y_translation #il faut ajouter de la position du haut de sphere -1 a chaque fois
                z = rayon*sin(phi) * sin(theta) + z_translation
                self.position.append((x, y, z))

        # add bottom vertex

        # add top / bottom triangles
        for i in range(0, n_slices):
            i0 = i + 2
            i1 = (i + 1) % n_slices + 2
            self.index.append((0, i1, i0))
            i0 = i + 1 + n_slices * (n_stacks - 2) + 1
            i1 = (i + 1) % n_slices + n_slices * (n_stacks - 2) + 2
            self.index.append((1, i0, i1))

        # add quads per stack / slice
        for j in range(0, n_stacks - 2):
            j0 = j * n_slices + 2
            j1 = (j + 1) * n_slices + 2
            for i in range(0, n_slices):
                i0 = j0 + i
                i1 = j0 + (i + 1) % n_slices
                i2 = j1 + (i + 1) % n_slices
                i3 = j1 + i
                self.index.append((i0, i1, i2))
                self.index.append((i0, i2, i3))

        color = []
        for i in range(0, len(self.position)):
            color.append((0.87, 0.8, 0.66)) #couleur du sable
        self.color = (0, 0, 0)
        attributes = dict(position=self.position, color=color)
        super().__init__(shader, attributes=attributes, index=self.index)

    def getPosition(self):
        return self.position
    def getIndex(self):
        return self.index

class TexturedForm(Textured):
    """ Simple first textured object """
    def __init__(self, shader, tex_file, position, index):
        # prepare texture modes cycling variables for interactive toggling
        self.wraps = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                            GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filters = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                              (GL.GL_LINEAR, GL.GL_LINEAR),
                              (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap, self.filter = next(self.wraps), next(self.filters)
        self.file = tex_file
        # mesh = Mesh(shader, attributes=dict(position=scaled), index=indices)
        mesh = Mesh(shader, attributes=dict(position=position, tex_coord=position), index=index)
        
        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        texture = Texture(tex_file, self.wrap, *self.filter)
        super().__init__(mesh, diffuse_map=texture)

    def key_handler(self, key):
        # cycle through texture modes on keypress of F6 (wrap) or F7 (filtering)
        self.wrap = next(self.wraps) if key == glfw.KEY_F6 else self.wrap
        self.filter = next(self.filters) if key == glfw.KEY_F7 else self.filter
        if key in (glfw.KEY_F6, glfw.KEY_F7):
            texture = Texture(self.file, self.wrap, *self.filter)
            self.textures.update(diffuse_map=texture)


class Skybox1(Mesh):
    def __init__(self, shader, file): 
       
        self.position = np.array(((-50, -50, -50), (50, -50, -50), (50, 50, -50), (-50, 50, -50),
                        (50, -50, 50), (-50, -50, 50), (-50, 50, 50), (50, 50, 50)), 'f')
        self.index = np.array((0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7), 'f')
        # , 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23), 'f')
        self.color = (1, 1, 1)
        self.attributes = dict(position=self.position, color=self.color)
        super().__init__(shader, attributes=self.attributes, index=self.index)
        self.cube = Node(transform=translate(25, -10, -10) @ scale(20, 20, 20))
        self.cube.add(TexturedForm(shader, file, self.position, self.index))
    
    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def getCarre(self):
        return self.cube
    def getIndex(self):
        return self.index

class Skybox2(Mesh):
    def __init__(self, shader, file):
        self.position = np.array(((-50, -50, -50), (-50, 50, -50), (-50, 50, 50), (-50, -50, 50),
                        (50, -50, -50), (50, -50, 50), (50, 50, 50), (50, 50, -50)), 'f')
        self.index = np.array((0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7), 'f')
        self.attributes = dict(position=self.position, color=(1, 1, 1))
        super().__init__(shader, attributes=self.attributes, index=self.index)
        self.cube = Node(transform=translate(25, -10, -10) @ scale(20, 20, 20))
        self.cube.add(TexturedForm(shader, file, self.position, self.index))
       
    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def getCarre(self):
        return self.cube
    def getIndex(self):
        return self.index

class Skybox3(Mesh):
    def __init__(self, shader, file): 
        self.position = np.array(((-50, 50, -50), (50, 50, -50), (50, 50, 50), (-50, 50, 50),
                        (-50, -50, 50), (50, -50, 50), (50, -50, -50), (-50, -50, -50)), 'f')
        self.index = np.array((0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7), 'f')
        self.attributes = dict(position=self.position, color=(1, 1, 1))
        super().__init__(shader, attributes=self.attributes, index=self.index)
        self.cube = Node(transform=translate(25, -10, -10) @ scale(20, 20, 20))
        self.cube.add(TexturedForm(shader, file, self.position, self.index))

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        super().draw(primitives=primitives, global_color=self.color, **uniforms)

    def getCarre(self):
        return self.cube
    def getIndex(self):
        return self.index         

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    shader = Shader("skinning.vert", "texture2.frag")
    shader2 = Shader("texture2.vert", "texture2.frag")
    shader3 = Shader("texture3.vert", "texture2.frag")
    shader4 = Shader("skinning.vert", "mer.frag")
    shader_soleil = Shader("skinning.vert", "texture2.frag")
    shader_sphere = Shader("texture2.vert", "texture2.frag")
    shader_dino = Shader("skinning.vert", "texture2.frag")
    shader_seagull = Shader("skinning.vert", "texture2.frag")
    shader_rogalic = Shader("skinning.vert", "texture2.frag")
    shader_hen = Shader("skinning.vert", "texture2.frag")
    shader_pointeur = Shader("skinning.vert", "texture2.frag")
    shader_arm = Shader("skinning.vert", "texture2.frag")
    shader_mannequin = Shader("phong.vert", "lambertian.frag")

    #light_dir = (10, -5, -10)
    light_dir = (0, -0.707, 0.707)


    if len(sys.argv) < 2:
        
        print('Usage:\n\t%s [3dfile]*\n\n3dfile\t\t the filename of a model in'
              ' format supported by assimp.' % (sys.argv[0],))
        # viewer.add(TexturedPlane(shader2, "rose-millenial.png", ((-2000, -2000, -2000), (2000, -2000, -2000), (2000, 2000, -2000), (-2000, 2000, -2000))))
        # viewer.add(TexturedPlane(shader2, "rose-millenial.png", ((-2000, -2000, -2000), (-2000, 2000, -2000), (-2000, 2000, 2000), (-2000, -2000, 2000))))
        # viewer.add(TexturedPlane(shader2, "rose-millenial.png", ((2000, -2000, 2000), (-2000, -2000, 2000), (-2000, 2000, 2000), (2000, 2000, 2000))))
        # viewer.add(TexturedPlane(shader2, "rose-millenial.png", ((2000, -2000, -2000), (2000, -2000, 2000), (2000, 2000, 2000), (2000, 2000, -2000))))
        # viewer.add(TexturedPlane(shader2, "rose-millenial.png", ((-2000, 2000, -2000), (2000, 2000, -2000), (2000, 2000, 2000), (-2000, 2000, 2000))))
        # viewer.add(TexturedPlane(shader2, "rose-millenial.png", ((-2000, -2000, 2000), (2000, -2000, 2000), (2000, -2000, -2000), (-2000, -2000, -2000))))

        mer_cube = Node(transform=translate(-163, -90, 163) @ scale(320, 91, 0))
        mer_cube.add(TexturedPlane(shader2, "mer.png", ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
        viewer.add(mer_cube)
        mer_cube = Node(transform=translate(157, -90, 163) @ rotate((0, 1, 0), 90) @ scale(320, 91, 0))
        mer_cube.add(TexturedPlane(shader2, "mer.png", ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
        viewer.add(mer_cube)
        mer_cube = Node(transform=translate(157, -90, -157) @ rotate((0, 1, 0), 180) @ scale(320, 91, 0))
        mer_cube.add(TexturedPlane(shader2, "mer.png", ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
        viewer.add(mer_cube)
        mer_cube = Node(transform=translate(-163, -90, -157) @ rotate((0, 1, 0), -90) @ scale(320, 91, 0))
        mer_cube.add(TexturedPlane(shader2, "mer.png", ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
        viewer.add(mer_cube)
        mer_cube = Node(transform=translate(-163, -90, -157) @ rotate((1, 0, 0), 90) @ scale(320, 320, 0))
        mer_cube.add(TexturedPlane(shader2, "mer.png", ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
        viewer.add(mer_cube)
        viewer.add(TexturedPlane(shader2, "mer.png", ((157, -90, -157), (-163, -90, -157), (-163, 1, -157), (157, 1, -157))))
        viewer.add(TexturedPlane(shader2, "mer.png", ((-163, -90, -157), (-163, -90, 163), (-163, 1, 163), (-163, 1, -157))))
        viewer.add(TexturedPlane(shader2, "mer.png", ((-163, -90, -157), (157, -90, -157), (157, -90, 163), (-163, -90, 163))))
        


        """ ------------------------- Définition de nos objets ---------------------------------------------------- """ 
        skybox = Skybox1(shader, 'skybox.png')
        # cube = Node(transform=translate(25, -10, -10) @ scale(100, 100, 100))
        # cube.add(TexturedForm(shader_sphere, "skybox.png", skybox.getPosition(), skybox.getIndex()))
        viewer.add(skybox.getCarre())
        skybox = Skybox2(shader, 'skybox.png')
        viewer.add(skybox.getCarre())
        skybox = Skybox3(shader, 'skybox.png')
        viewer.add(skybox.getCarre())


        mer = Water(transform=translate(-3, 0, 3) @ scale(80, 20, 80))
        mer.add(*load("Ocean/Ocean.obj", shader4, light_dir=light_dir))
        viewer.add(mer)
        

        sphere = Sphere2(shader_sphere, 100, 40, 5, -100, -10, 50)
        soleil = Node(transform=translate(100, 300, -500) @ scale(5, 5, 5))
        soleil.add(TexturedForm(shader_sphere, "soleil.png", sphere.getPosition(), sphere.getIndex()))
        # soleil.add(*load("wooden_sphere.obj", shader_soleil, light_dir=light_dir))
        viewer.add(soleil)
        
        central_island_1 = Node(transform=translate(-60, -10, -40) @ scale(4, 4, 4))
        central_island_1.add(*load("central_Island/Groupofpalms.obj", shader, light_dir=light_dir))
        viewer.add(central_island_1)
        central_island_2 = Node(transform=translate(60, -10, 40) @ scale(4, 4, 4))
        central_island_2.add(*load("central_Island/Groupofpalms.obj", shader, light_dir=light_dir))
        viewer.add(central_island_2)
        
        sphere = Sphere2(shader_sphere, 100, 40, 50, -90, -10, 100)
        island = Node(transform=translate(25, -10, -10))
        island.add(TexturedForm(shader_sphere, "sand.png", sphere.getPosition(), sphere.getIndex()))
        viewer.add(island)

        sphere = Sphere2(shader_sphere, 100, 40, 35, -40, -10, 100)
        island = Node(transform=translate(25, -10, -10))
        island.add(TexturedForm(shader_sphere, "sand.png", sphere.getPosition(), sphere.getIndex()))
        viewer.add(island)

        sphere = Sphere2(shader_sphere, 100, 40, 40, -100, -10, 50)
        island = Node(transform=translate(-15, -10, -10))
        island.add(TexturedForm(shader_sphere, "sand.png", sphere.getPosition(), sphere.getIndex()))
        viewer.add(island)
            
            
        arm = Node(transform=translate(-15, 22, 50) @ rotate((0, 0, 1), 45) @ scale(1, 1, 1))
        arm.add(SkinnedCylinder(shader_arm))
        viewer.add(arm)

        viewer.add(Mannequin(shader_mannequin, light_dir, transform=translate(-41, 29, 100) @ rotate((0, 1, 0), 180) @ scale(.3, .3, .3)))

    
        tree2 = Node(transform=translate(-90, 40, 120) @ scale(0.5, 0.5, 0.5))
        tree2.add(*load("FantasyWorld/NatureAssets/Tree_03.FBX", shader, light_dir=light_dir)) #, tex_file='FantasyWorld/NatureAssets/Textures/Nature_Atlas_1.tga'))
        viewer.add(tree2)

        hen = Node(transform= translate(28, -2.6, 20) @ rotate((1, 0, 0), 45) @ scale(0.5, 0.5, 0.5))
        hen.add(*load("FantasyWorld/Animated/Hen/hen.FBX", shader_hen, light_dir=light_dir))
        viewer.add(hen)

        dino = Node(transform=translate(55, 5, 20) @ scale(0.3, 0.3, 0.3))
        dino.add(*load("FantasyCharacters/Dino/Dino_attack_1.fbx", shader_dino, light_dir=light_dir))
        viewer.add(dino)

        rogalic = Node(transform= translate(-30, 19, 72) @ scale(0.3, 0.3, 0.3))
        rogalic.add(*load("FantasyCharacters/Rogalic/Rogalic_attack_1.fbx", shader_rogalic, light_dir=light_dir))
        viewer.add(rogalic)
        
        bridge = Node(transform= translate(-470, -2, -855) @ scale(0.5, 0.5, 0.5))
        bridge.add(*load("FantasyWorld/Constructed/Constructed_BridgeWood02.FBX", shader2, light_dir=light_dir, tex_file="FantasyWorld/Constructed/Textures/All_Assets.tga"))
        viewer.add(bridge)
        
        
        for i in range(3):
            rock = Node(transform=translate(70 + i*10, -2.5, -200) @ scale(0.4, 0.4, 0.4))
            rock.add(*load("FantasyWorld/NatureAssets/Rock_01.FBX", shader, light_dir=light_dir, tex_file='FantasyWorld/NatureAssets/Textures/Nature_Atlas_1.tga'))
            viewer.add(rock)
            viewer.add(*[mesh for file in sys.argv[1:]
                    for mesh in load(file, shader, light_dir=light_dir, tex_file='FantasyWorld/NatureAssets/Textures/Nature_Atlas_1.tga')])

            house_mush = Node(transform=translate(-50 - i*6, 5, 25) @ scale(0.2, 0.2, 0.2))
            house_mush.add(*load("FantasyWorld/Constructable_Elements/HouseMushroom.FBX", shader, light_dir=light_dir, tex_file='FantasyWorld/NatureAssets/Textures/Nature_Atlas_1.tga'))
            viewer.add(house_mush)
            
        for i in range(1, 6):
            tree1 = Node(transform=translate(-90, 35-i*2 , 120-i*15) @ scale(0.5, 0.5, 0.5))
            tree1.add(*load("FantasyWorld/NatureAssets/Tree_0{}.FBX".format(i), shader, light_dir=light_dir)) #, tex_file='FantasyWorld/NatureAssets/Textures/Nature_Atlas_1.tga'))
            viewer.add(tree1)

        N=10
        for i in range(0, N):
            angle = 2*i*np.pi / N 
            mother_tree = Node(transform=translate(-10 + 30*np.cos(angle), 1, 10 + 30*np.sin(angle)) @ scale(0.05, 0.05, 0.05))
            mother_tree.add(*load("FantasyWorld/NatureAssets/Mother_Tree.FBX", shader))
            viewer.add(mother_tree)

        """ ------------------------- Animations de nos objets ---------------------------------------------------- """ 

        for i in range(6):
            angle = 2*i*np.pi / 10
            seagull = Node(transform=translate(10 + 30*np.cos(angle), 20, 100 + 30*np.sin(angle)) @ rotate((1, 0, 0), angle=45) @ scale(0.8, 0.8, 0.8))
            seagull.add(*load("FantasyWorld/Animated/Seagull/seagul.FBX", shader_seagull, light_dir=light_dir))
            transkey, rotkey, scalekey = sens_rotation(-1, 0, 'seagull')
            keynode = KeyFrameControlNode(transkey, rotkey, scalekey, 'seagull')         
            keynode.add(seagull)
            viewer.add(keynode)


        pointeur = Node(transform=translate(100, 30, 40) @ scale(1, 1, 1))
        pointeur.add(*load("FantasyWorld/NatureAssets/Crystal_05.FBX", shader_pointeur, light_dir=light_dir))
        transkey, rotkey, scalekey = sens_rotation(1, 0, 'pointeur', move=False)
        keynode = KeyFrameControlNode(transkey, rotkey, scalekey, 'pointeur')         
        keynode.add(pointeur)
        viewer.add(keynode)


        boat = Node(transform=translate(-10, -1, -40) @ scale(0.3, 0.3, 0.3))
        boat.add(*load("FantasyWorld/Boats/Galleon.FBX", shader, light_dir=light_dir, tex_file='FantasyWorld/Boats/Textures/Ships_1.tga'))
        transkey, rotkey, scalekey = sens_rotation(1, -180, 'boat')
        keynode = KeyFrameControlNode(transkey, rotkey, scalekey)         
        keynode.add(boat)
        viewer.add(keynode)

        
    else:
        #viewer.add(*[mesh for file in sys.argv[1:]
        #             for mesh in load(file, shader_mannequin, light_dir=light_dir, tex_file='FantasyWorld/NatureAssets/Textures/Nature_Atlas_1.tga')])
        mannequin = Mannequin(shader_mannequin, light_dir=(0,0,-1))
        mannequin.pousse()
        viewer.add(mannequin)
    
        
    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    main()                     # main function keeps variables locally scoped
