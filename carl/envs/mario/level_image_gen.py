# Code from https://github.com/Mawiszus/TOAD-GAN
import os

from PIL import Image, ImageOps, ImageEnhance


class LevelImageGen:
    """ Generates PIL Image files from Super Mario Bros. ascii levels.
    Initialize once and then use LevelImageGen.render() to generate images. """
    def __init__(self, sprite_path):
        """ sprite_path: path to the folder of sprite files, e.g. 'mario/sprites/' """

        # Load Graphics (assumes sprite_path points to "img" folder of Mario-AI-Framework or provided sprites folder
        mariosheet = Image.open(os.path.join(sprite_path, 'smallmariosheet.png'))
        enemysheet = Image.open(os.path.join(sprite_path, 'enemysheet.png'))
        itemsheet = Image.open(os.path.join(sprite_path, 'itemsheet.png'))
        mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))

        # Cut out the actual sprites:
        sprite_dict = dict()
        # Mario Sheet
        sprite_dict['M'] = mariosheet.crop((4*16, 0, 5*16, 16))

        # Enemy Sheet
        enemy_names = ['r', 'k', 'g', 'y', 'wings', '*', 'plant']
        for i, e in enumerate(enemy_names):
            sprite_dict[e] = enemysheet.crop((0, i*2*16, 16, (i+1)*2*16))

        sprite_dict['E'] = enemysheet.crop((16, 2*2*16, 2*16, 3*2*16))  # Set generic enemy to second goomba sprite
        sprite_dict['plant'] = enemysheet.crop((16, (len(enemy_names)-1)*2*16, 2*16, len(enemy_names)*2*16))

        # Item Sheet
        sprite_dict['shroom'] = itemsheet.crop((0, 0, 16, 16))
        sprite_dict['flower'] = itemsheet.crop((16, 0, 2*16, 16))
        sprite_dict['flower2'] = itemsheet.crop((0, 16, 16, 2*16))
        sprite_dict['1up'] = itemsheet.crop((16, 16, 2*16, 2*16))

        # Map Sheet
        map_names = ['-', 'X', '#', 'B', 'b', 'b2', 'S', 'L',
                     '?', 'dump', '@', 'Q', 'dump', '!', 'D', 'o',
                     'o2', 'o3', '<', '>', '[', ']', 'bg_sl_l', 'bg_top',
                     'bg_sl_r', 'bg_m_l', 'bg_m', 'bg_m_r', 'bush_l', 'bush_m', 'bush_r', 'cloud_l',
                     'cloud_m', 'cloud_r', 'cloud_b_l', 'cloud_b_m', 'cloud_b_r', 'waves', 'water', 'F_top',
                     'F_b', 'F', 'bg_sky', '%', '%_l', '%_r', '%_m', '|',
                     '1', '2', 'C', 'U', 'T', 't', 'dump', 'dump']

        sheet_length = (7, 8)
        sprite_counter = 0
        for i in range(sheet_length[0]):
            for j in range(sheet_length[1]):
                sprite_dict[map_names[sprite_counter]] = mapsheet.crop((j*16, i*16, (j+1)*16, (i+1)*16))
                sprite_counter += 1

        sprite_dict['@'] = sprite_dict['?']
        sprite_dict['!'] = sprite_dict['Q']

        self.sprite_dict = sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """ Helper to make correct sprites and sprite sizes to draw into the image.
         Some sprites are bigger than one tile and the renderer needs to adjust for them."""

        # Init default size
        new_left = curr_x * 16
        new_top = curr_y * 16
        new_right = (curr_x + 1) * 16
        new_bottom = (curr_y + 1) * 16

        # Handle sprites depending on their type:
        if sprite_key == 'F':  # Flag Pole
            actual_sprite = Image.new('RGBA', (2*16, curr_y*16))
            actual_sprite.paste(self.sprite_dict['F_top'], (16, 0, 2*16, 16))
            for s in range(curr_y):
                actual_sprite.paste(self.sprite_dict['F_b'], (16, (s+1)*16, 2*16, (s+2)*16))
            actual_sprite.paste(self.sprite_dict['F'], (7, 1*16, 16 + 7, 2*16))
            new_left = new_left - 16
            new_top = new_top - (curr_y-1)*16

        elif sprite_key in ['y', 'E', 'g', 'k', 'r']:  # enemy sprite
            actual_sprite = self.sprite_dict[sprite_key]
            new_top = new_top-16

        elif sprite_key in ['Y', 'K', 'R']:  # winged spiky/koopa sprite
            actual_sprite = Image.new('RGBA', (2 * 16, 2 * 16))
            actual_sprite.paste(self.sprite_dict[str.lower(sprite_key)], (16, 0, 2*16, 2*16))
            actual_sprite.paste(self.sprite_dict['wings'], (7, -7, 16+7, 2*16-7))
            new_left = new_left-16
            new_top = new_top-16

        elif sprite_key == 'G':  # winged goomba sprite (untested because original has none?)
            actual_sprite = Image.new('RGBA', (3 * 16, 2 * 16))
            actual_sprite.paste(self.sprite_dict['wings'], (1, -5, 16+1, 2*16-5))
            actual_sprite.paste(ImageOps.mirror(self.sprite_dict['wings']), (2*16-1, -5, 3*16-1, 2*16-5))
            actual_sprite.paste(self.sprite_dict[str.lower(sprite_key)], (16, 0, 2*16, 2*16))
            new_left = new_left-16
            new_top = new_top-16
            new_right = new_right+16

        elif sprite_key == '%':  # jump through platform
            if curr_x == 0:
                if len(ascii_level[curr_y]) > 1 and ascii_level[curr_y][curr_x+1] == sprite_key:  # middle piece
                    actual_sprite = self.sprite_dict['%_m']
                else:  # single_piece
                    actual_sprite = self.sprite_dict['%']
            elif ascii_level[curr_y][curr_x-1] == sprite_key:
                if curr_x >= (len(ascii_level[curr_y]) - 1):  # right end piece
                    actual_sprite = self.sprite_dict['%_r']
                elif ascii_level[curr_y][curr_x+1] == sprite_key:  # middle piece
                    actual_sprite = self.sprite_dict['%_m']
                else:  # right end piece
                    actual_sprite = self.sprite_dict['%_r']
            else:
                if curr_x >= (len(ascii_level[curr_y]) - 1):  # single piece
                    actual_sprite = self.sprite_dict['%']
                elif ascii_level[curr_y][curr_x+1] == sprite_key:  # left end piece
                    actual_sprite = self.sprite_dict['%_l']
                else:  # single piece
                    actual_sprite = self.sprite_dict[sprite_key]

        elif sprite_key == 'b':  # bullet bill tower
            if curr_y > 0:
                if ascii_level[curr_y-1][curr_x] == sprite_key:
                    actual_sprite = self.sprite_dict['b2']
                else:
                    actual_sprite = self.sprite_dict[sprite_key]
            else:
                actual_sprite = self.sprite_dict[sprite_key]

        elif sprite_key == '*':  # alternative bullet bill tower
            if curr_y > 0:
                if ascii_level[curr_y-1][curr_x] != sprite_key:  # top
                    actual_sprite = self.sprite_dict['B']
                elif curr_y > 1:
                    if ascii_level[curr_y-2][curr_x] != sprite_key:
                        actual_sprite = self.sprite_dict['b']
                    else:
                        actual_sprite = self.sprite_dict['b2']
            else:
                actual_sprite = self.sprite_dict['b2']

        elif sprite_key in ['T', 't']:  # Pipes

            # figure out what kind of pipe this is
            if curr_y > 0 and ascii_level[curr_y-1][curr_x] == sprite_key:
                is_top = False
            else:
                is_top = True

            pipelength_t = 0
            while curr_y - pipelength_t >= 0 and ascii_level[curr_y - pipelength_t][curr_x] == sprite_key:
                pipelength_t += 1

            pipelength_b = 0
            while curr_y + pipelength_b < len(ascii_level) and ascii_level[curr_y + pipelength_b][curr_x] == sprite_key:
                pipelength_b += 1

            pipelength_l = 0
            while curr_x - pipelength_l >= 0 and ascii_level[curr_y][curr_x - pipelength_l] == sprite_key:
                pipelength_l += 1

            pipelength_r = 0
            while curr_x + pipelength_r < len(ascii_level[curr_y]) and ascii_level[curr_y][curr_x - pipelength_r] == sprite_key:
                pipelength_r += 1

            # Check for fall out criteria
            try:
                if pipelength_l % 2 == 0:  # second half of a double pipe
                    is_left = False
                    is_right = True
                elif pipelength_l % 2 == 1:
                    if curr_x >= len(ascii_level[curr_y]) or ascii_level[curr_y][curr_x + 1] != sprite_key:
                        is_left = False
                        is_right = False
                    else:
                        is_left = True
                        is_right = False
                else:
                    is_left = False
                    is_right = False

                if is_left:
                    if ascii_level[curr_y - pipelength_t][curr_x + 1] == sprite_key:
                        is_left = False
                        is_right = False
                    if ascii_level[curr_y - pipelength_t + 1][curr_x + 1] != sprite_key:
                        is_left = False
                        is_right = False
                if is_right:
                    if ascii_level[curr_y - pipelength_t][curr_x - 1] == sprite_key:
                        is_left = False
                        is_right = False
                    if ascii_level[curr_y - pipelength_t + 1][curr_x - 1] != sprite_key:
                        is_left = False
                        is_right = False
                if curr_y + pipelength_b < len(ascii_level):
                    if is_left:
                        if ascii_level[curr_y + pipelength_b][curr_x + 1] == sprite_key:
                            is_left = False
                            is_right = False
                        if ascii_level[curr_y + pipelength_b - 1][curr_x + 1] != sprite_key:
                            is_left = False
                            is_right = False
                    if is_right:
                        if ascii_level[curr_y + pipelength_b][curr_x - 1] == sprite_key:
                            is_left = False
                            is_right = False
                        if ascii_level[curr_y + pipelength_b - 1][curr_x - 1] != sprite_key:
                            is_left = False
                            is_right = False
            except IndexError:
                # Default to single pipe
                is_left = False
                is_right = False

            if is_top:
                if is_left:
                    actual_sprite = self.sprite_dict['<']
                elif is_right:
                    if sprite_key == 'T':
                        actual_sprite = Image.new('RGBA', (2 * 16, 3 * 16))
                        actual_sprite.paste(self.sprite_dict['plant'], (8, 5, 16 + 8, 2 * 16 + 5))
                        actual_sprite.paste(self.sprite_dict['<'], (0, 2 * 16, 16, 3 * 16))
                        actual_sprite.paste(self.sprite_dict['>'], (16, 2 * 16, 2 * 16, 3 * 16))
                        new_left = new_left - 16
                        new_top = new_top - 2 * 16
                    else:
                        actual_sprite = self.sprite_dict['>']
                else:
                    if sprite_key == 'T':
                        actual_sprite = Image.new('RGBA', (16, 3 * 16))
                        actual_sprite.paste(self.sprite_dict['plant'], (0, 5, 16, 2 * 16 + 5))
                        actual_sprite.paste(self.sprite_dict['T'], (0, 2 * 16, 16, 3 * 16))
                        new_top = new_top - 2 * 16
                    else:
                        actual_sprite = self.sprite_dict['T']
            else:
                if is_left:
                    actual_sprite = self.sprite_dict['[']
                elif is_right:
                    actual_sprite = self.sprite_dict[']']
                else:
                    actual_sprite = self.sprite_dict['t']

        elif sprite_key in ['?', '@', 'Q', '!', 'C', 'U', 'L']:  # Block/Brick hidden items
            if sprite_key == 'L':
                i_key = '1up'
            elif sprite_key in ['?', '@', 'U']:
                i_key = 'shroom'
            else:
                i_key = 'o'

            mask = self.sprite_dict[i_key].getchannel(3)
            mask = ImageEnhance.Brightness(mask).enhance(0.7)
            actual_sprite = Image.composite(self.sprite_dict[i_key], self.sprite_dict[sprite_key], mask=mask)

        elif sprite_key in ['1', '2']:  # Hidden block
            if sprite_key == '1':
                i_key = '1up'
            else:
                i_key = 'o'

            mask1 = self.sprite_dict['D'].getchannel(3)
            mask1 = ImageEnhance.Brightness(mask1).enhance(0.5)
            tmp_sprite = Image.composite(self.sprite_dict['D'], self.sprite_dict[sprite_key], mask=mask1)
            mask = self.sprite_dict[i_key].getchannel(3)
            mask = ImageEnhance.Brightness(mask).enhance(0.7)
            actual_sprite = Image.composite(self.sprite_dict[i_key], tmp_sprite, mask=mask)

        else:
            actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """ Renders the ascii level as a PIL Image. Assumes the Background is sky """
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)

        # Fill base image with sky tiles
        dst = Image.new('RGB', (len_level*16, height_level*16))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(self.sprite_dict['bg_sky'], (x*16, y*16, (x+1)*16, (y+1)*16))

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[y][x]
                sprite, box = self.prepare_sprite_and_box(ascii_level, curr_sprite, x, y)
                dst.paste(sprite, box, mask=sprite)

        return dst
