import numpy as np

import PIL
from PIL import Image

import matplotlib.pyplot as plt

object_prompts_list = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} with a wheat field in the background',
    'a {0} {1} with a tree and autumn leaves in the background',
    'a {0} {1} with the Eiffel Tower in the background',
    'a {0} {1} floating on top of water',
    'a {0} {1} floating in an ocean of milk',
    'a {0} {1} on top of green grass with sunflowers around it',
    'a {0} {1} on top of a mirror',
    'a {0} {1} on top of the sidewalk in a crowded street',
    'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a white rug',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}',
]

live_prompts_list = [
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on the beach',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} wearing a red hat',
    'a {0} {1} wearing a santa hat',
    'a {0} {1} wearing a rainbow scarf',
    'a {0} {1} wearing a black top hat and a monocle',
    'a {0} {1} in a chef outfit',
    'a {0} {1} in a firefighter outfit',
    'a {0} {1} in a police outfit',
    'a {0} {1} wearing pink glasses',
    'a {0} {1} wearing a yellow shirt',
    'a {0} {1} in a purple wizard outfit',
    'a red {0} {1}',
    'a purple {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a cube shaped {0} {1}',
]

eval_prompts_list = [
    'a cube shaped {0} {1}',
    'a purple {0} {1}',
    'a red {0} {1}',
    'a shiny {0} {1}',
    'a wet {0} {1}',
    'a {0} {1} floating in an ocean of milk',
    'a {0} {1} floating on top of water',
    'a {0} {1} in a chef outfit',
    'a {0} {1} in a firefighter outfit',
    'a {0} {1} in a police outfit',
    'a {0} {1} in a purple wizard outfit',
    'a {0} {1} in the jungle',
    'a {0} {1} in the snow',
    'a {0} {1} on a cobblestone street',
    'a {0} {1} on the beach',
    'a {0} {1} on top of a dirt road',
    'a {0} {1} on top of a mirror',
    'a {0} {1} on top of a purple rug in a forest',
    'a {0} {1} on top of a white rug',
    'a {0} {1} on top of a wooden floor',
    'a {0} {1} on top of green grass with sunflowers around it',
    'a {0} {1} on top of pink fabric',
    'a {0} {1} on top of the sidewalk in a crowded street',
    'a {0} {1} wearing a black top hat and a monocle',
    'a {0} {1} wearing a rainbow scarf',
    'a {0} {1} wearing a red hat',
    'a {0} {1} wearing a santa hat',
    'a {0} {1} wearing a yellow shirt',
    'a {0} {1} wearing pink glasses',
    'a {0} {1} with a blue house in the background',
    'a {0} {1} with a city in the background',
    'a {0} {1} with a mountain in the background',
    'a {0} {1} with a tree and autumn leaves in the background',
    'a {0} {1} with a wheat field in the background',
    'a {0} {1} with the Eiffel Tower in the background'
]

training_prompts_list = [
    'a pyramid shaped {0} {1}',
    'a yellow {0} {1}',
    'a green {0} {1}',
    'a dirty {0} {1}',
    'a bright {0} {1}',
    'a {0} {1} floating in an lake of juice',
    'a {0} {1} floating on top of grass',
    'a {0} {1} in a doctor outfit',
    'a {0} {1} in a scientist outfit',
    'a {0} {1} in a postman outfit',
    'a {0} {1} in a blue farmer outfit',
    'a {0} {1} in the savanna',
    'a {0} {1} in the sky',
    'a {0} {1} in a dark alley',
    'a {0} {1} in a lush forest',
    'a {0} {1} on top of a black and white checkered tile floor',
    'a {0} {1} on top of a stack of books in a cozy library',
    'a {0} {1} on top of a red and white checkered picnic blanket',
    'a {0} {1} on top of a silver metal bench',
    'a {0} {1} on top of a blue velvet pillow',
    'a {0} {1} on top of a black leather couch',
    'a {0} {1} on top of a concrete floor with graffiti',
    'a {0} {1} on top of a snowy mountain peak',
    'a {0} {1} wearing a green bow tie',
    'a {0} {1} carrying a brown leather messenger bag',
    'a {0} {1} wearing a blue bandana and aviator sunglasses',
    'a {0} {1} wearing a white lab coat',
    'a {0} {1} with a pink umbrella',
    'a {0} {1} wearing a red hoodie',
    'a {0} {1} with a sunset and palm trees in the background',
    'a {0} {1} with a city skyline at night in the background',
    'a {0} {1} with a castle in the background',
    'a {0} {1} with a forest in the background',
    'a {0} {1} with the Sydney Opera House in the background',
    'a {0} {1} with a rocky coastline in the background',
]

classes_data = {
    'person':                 ('sks', '<person>',         'person'        ),    # noqa
    'mini_person1':           ('sks', '<person>',         'person'        ),    # noqa
    'mini_person2':           ('sks', '<person>',         'person'        ),    # noqa
    'mini_person3':           ('sks', '<person>',         'person'        ),    # noqa
    'mini_person':            ('sks', '<person>',         'person'        ),    # noqa
    'me':                     ('sks', '<me-person>',      'person'        ),    # noqa
    'backpack':               ('sks', '<backpack>',       'backpack'      ),    # noqa
    'mini_backpack':          ('sks', '<backpack>',       'backpack'      ),    # noqa
    'backpack1':              ('sks', '<backpack>',       'backpack'      ),    # noqa
    'backpack_dog':           ('sks', '<backpack>',       'backpack'      ),    # noqa
    'bear_plushie':           ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
    'berry_bowl':             ('sks', '<bowl>',           'bowl'          ),    # noqa
    'can':                    ('sks', '<can>',            'can'           ),    # noqa
    'candle':                 ('sks', '<candle>',         'candle'        ),    # noqa
    'cat':                    ('sks', '<cat>',            'cat'           ),    # noqa
    'cat2':                   ('sks', '<cat>',            'cat'           ),    # noqa
    'clock':                  ('sks', '<clock>',          'clock'         ),    # noqa
    'colorful_sneaker':       ('sks', '<sneaker>',        'sneaker'       ),    # noqa
    'dog':                    ('sks', '<dog>',            'dog'           ),    # noqa
    'dog2':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog3':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog5':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog6':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'mini_dog6':              ('sks', '<dog>',            'dog'           ),    # noqa
    'dog7':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog8':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'duck_toy':               ('sks', '<toy>',            'toy'           ),    # noqa
    'fancy_boot':             ('sks', '<boot>',           'boot'          ),    # noqa
    'grey_sloth_plushie':     ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
    'monster_toy':            ('sks', '<toy>',            'toy'           ),    # noqa
    'pink_sunglasses':        ('sks', '<glasses>',        'glasses'       ),    # noqa
    'poop_emoji':             ('sks', '<toy>',            'toy'           ),    # noqa
    'rc_car':                 ('sks', '<toy>',            'toy'           ),    # noqa
    'red_cartoon':            ('sks', '<cartoon>',        'cartoon'       ),    # noqa
    'robot_toy':              ('sks', '<toy>',            'toy'           ),    # noqa
    'mini_robot_toy':         ('sks', '<toy>',            'toy'           ),    # noqa
    'shiny_sneaker':          ('sks', '<sneaker>',        'sneaker'       ),    # noqa
    'teapot':                 ('sks', '<teapot>',         'teapot'        ),    # noqa
    'vase':                   ('sks', '<vase>',           'vase'          ),    # noqa
    'wolf_plushie':           ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
}

live_object_data = {
    'person':                 'live',    # noqa
    'backpack':               'object',    # noqa
    'stuffed animal':         'object',    # noqa
    'bowl':                   'object',    # noqa
    'can':                    'object',    # noqa
    'candle':                 'object',    # noqa
    'cat':                    'live',    # noqa
    'clock':                  'object',    # noqa
    'sneaker':                'object',    # noqa
    'dog':                    'live',    # noqa
    'toy':                    'object',    # noqa
    'boot':                   'object',    # noqa
    'sunglasses':             'object',    # noqa
    'cartoon':                'object',    # noqa
    'teapot':                 'object',    # noqa
    'vase':                   'object',    # noqa
} | {f'person{_}': 'live' for _ in range(1, 51)}

best_imgs = {
    'person':                 '01.png',    # noqa
    'person1':                'seed0000.png',     # noqa
    'person2':                'seed0002.png',     # noqa
    'person3':                'seed0004.png',     # noqa
    'person4':                'seed0005.png',     # noqa
    'person5':                'seed0009.png',     # noqa
    'person6':                'seed0010.png',     # noqa
    'person7':                'seed0011.png',     # noqa
    'person8':                'seed0012.png',     # noqa
    'person9':                'seed0013.png',     # noqa
    'person10':               'seed0014.png',     # noqa
    'person11':               'seed0015.png',     # noqa
    'person12':               'seed0016.png',     # noqa
    'person13':               'seed0017.png',     # noqa
    'person14':               'seed0019.png',     # noqa
    'person15':               'seed0020.png',     # noqa
    'person16':               'seed0022.png',     # noqa
    'person17':               'seed0023.png',     # noqa
    'person18':               'seed0024.png',     # noqa
    'person19':               'seed0025.png',     # noqa
    'person20':               'seed0026.png',     # noqa
    'person21':               'seed0027.png',     # noqa
    'person22':               'seed0031.png',     # noqa
    'person23':               'seed0032.png',     # noqa
    'person24':               'seed0035.png',     # noqa
    'person25':               'seed0036.png',     # noqa
    'person26':               'seed0037.png',     # noqa
    'person27':               'seed0041.png',     # noqa
    'person28':               'seed0042.png',     # noqa
    'person29':               'seed0043.png',     # noqa
    'person30':               'seed0044.png',     # noqa
    'person31':               'seed0045.png',     # noqa
    'person32':               'seed0046.png',     # noqa
    'person33':               'seed0047.png',     # noqa
    'person34':               'seed0048.png',     # noqa
    'person35':               'seed0049.png',     # noqa
    'person36':               'seed0050.png',     # noqa
    'person37':               'seed0051.png',     # noqa
    'person38':               'seed0052.png',     # noqa
    'person39':               'seed0053.png',     # noqa
    'person40':               'seed0054.png',     # noqa
    'person41':               'seed0055.png',     # noqa
    'person42':               'seed0056.png',     # noqa
    'person43':               'seed0058.png',     # noqa
    'person44':               'seed0060.png',     # noqa
    'person45':               'seed0064.png',     # noqa
    'person46':               'seed0066.png',     # noqa
    'person47':               'seed0067.png',     # noqa
    'person48':               'seed0068.png',     # noqa
    'person49':               'seed0069.png',     # noqa
    'person50':               'seed0070.png',     # noqa
    'backpack':               '02.jpg',    # noqa
    'backpack_dog':           '00.jpg',    # noqa
    'bear_plushie':           '03.jpg',    # noqa
    'berry_bowl':             '00.jpg',    # noqa
    'can':                    '04.jpg',    # noqa
    'candle':                 'object',    # noqa
    'cat':                    '03.jpg',    # noqa
    'cat2':                   '00.jpg',    # noqa
    'clock':                  '03.jpg',    # noqa
    'colorful_sneaker':       '01.jpg',    # noqa
    'dog':                    '02.jpg',    # noqa
    'dog2':                   '02.jpg',    # noqa
    'dog3':                   '00.jpg',    # noqa
    'dog5':                   '03.jpg',    # noqa
    'dog6':                   '02.jpg',    # noqa
    'dog7':                   '01.jpg',    # noqa
    'dog8':                   '01.jpg',    # noqa
    'duck_toy':               '01.jpg',    # noqa
    'fancy_boot':             '02.jpg',    # noqa
    'grey_sloth_plushie':     '01.jpg',    # noqa
    'monster_toy':            '00.jpg',    # noqa
    'pink_sunglasses':        '05.jpg',    # noqa
    'poop_emoji':             '00.jpg',    # noqa
    'rc_car':                 '03.jpg',    # noqa
    'red_cartoon':            '01.jpg',    # noqa
    'robot_toy':              '01.jpg',    # noqa
    'shiny_sneaker':          '04.jpg',    # noqa
    'teapot':                 '00.jpg',    # noqa
    'vase':                   '02.jpg',    # noqa
    'wolf_plushie':           'object',    # noqa
}

_LOAD_IMAGE_BACKENDS = {
    'PIL': lambda path: np.asarray(PIL.Image.open(path).convert('RGB')),
    'plt': lambda path: plt.imread(path),
    None: None,
}
_LOAD_IMAGE_BACKEND = _LOAD_IMAGE_BACKENDS['PIL']
