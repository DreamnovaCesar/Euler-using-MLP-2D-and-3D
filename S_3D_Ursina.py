from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

import time

app = Ursina()

block_pick = 1

def update():

    global block_pick

    if held_keys['1']: block_pick = 1
    if held_keys['2']: block_pick = 2
    if held_keys['3']: block_pick = 3
    if held_keys['4']: block_pick = 4

class Test_cube(Entity):
    def __init__(self):
        super().__init__(
            model = 'cube',
            color = color.white,
            texture = 'white_cube',
            rotation = Vec3(45, 45, 45)
        )

class Test_button(Button):
    def __init__(self):
        super().__init__(
            parent = scene,
            model = 'cube',
            texture = 'brick',
            color = color.blue,
            highlight_color = color.red,
            pressed_color = color.lime
        )

    def input(self, key):
        if self.hovered:
            if key == 'left mouse down':
                print('button pressed')

class Voxel(Button):
    def __init__(self, position = (0, 0, 0), color_ = color.white):

        self.Voxel = 1

        super().__init__(
            parent = scene,
            position = position,
            model = 'cube',
            origin_y = 0.5,
            texture = 'white_cube',
            color = color_,
            highlight_color = color.lime
        )


    def input(self, key):

        if self.hovered:

            if key == 'left mouse down':


                if block_pick == 1: 
                    voxel = Voxel(position = self.position + mouse.normal, color_ = color.red) 
                    Text_entity = Text('Color', scale = 2, color = color.red, origin = (-4, 3))

                if block_pick == 2: 
                    voxel = Voxel(position = self.position + mouse.normal, color_ = color.blue)
                    Text_entity = Text('Color', scale = 2, color = color.blue, origin = (-4, 4))
                    
                if block_pick == 3: 
                    voxel = Voxel(position = self.position + mouse.normal, color_ = color.green)
                    Text_entity = Text('Color', scale = 2, color = color.green, origin = (-4, 5))

                if block_pick == 4: 
                    voxel = Voxel(position = self.position + mouse.normal, color_ = color.white)
                    Text_entity = Text('Color', scale = 2, color = color.white, origin = (-4, 6))


            if key == 'right mouse down':
                destroy(self)

class Stats(Text):
    def __init__(self, Text_, Color_ = color.white):
        
        self.Text = Text_
        self.Color = Color_

    def create_text():
        Text_entity = Text('Color', scale = 2, color = color.green, origin = (-4, 5))

for z in range(10):
    for x in range(10):
        voxel = Voxel((x, 0, z))
        Total_voxels += voxel.Voxel

player = FirstPersonController()


app.run()

