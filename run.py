import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QApplication, QGridLayout, QHBoxLayout
from matplotlib.pyplot import draw
from constants import semantic_sensor_40cat, roomidx2name, room_set, objs_set
import os.path as osp
import copy
import os
import jsonlines
from PIL.ImageQt import ImageQt
import time
import gzip
import json
import random
import string
random.seed(0)
from utils.map_tools import get_maps, simloc2maploc, draw_agent, draw_point, draw_path, get_possible_paths, colorize_nav_map
from utils.instance_process import get_surrounding_objs, draw_instances, get_room_compass, draw_room_instances
from PIL import Image
import numpy as np
recolor_room = np.array(
            [[255, 255, 255, 0], [255, 128, 0, 200]], dtype=np.uint8
        )
recolor_object = np.array(
            [[255, 255, 255, 0], [255, 128, 0, 200]], dtype=np.uint8
        )
TOT_SAMPLES=20
direction_list={'forward', 'straight', 'turn', 'left', 'right', 'stop', 'wait'}

class ImageViewer(QWidget):

    def __init__(self, annt_root, split, merged = True):
        QWidget.__init__(self)
        self.merged = merged
        if self.merged:
            self.ordered_room_name = sorted({v:k for k,v in room_set.items()}.items(), key=lambda item: item[1])
            self.ordered_obj_name = sorted({v:k for k,v in objs_set.items()}.items(), key=lambda item: item[1])
        else:
            self.ordered_room_name = sorted(roomidx2name.items(), key=lambda item: item[1])
            self.ordered_obj_name = sorted(semantic_sensor_40cat.items(), key=lambda item: item[1])
        
        self.setup_ui()
        out_root = f'out/manual_eval_{int(time.time())}'
        os.makedirs(out_root, exist_ok=True)
        self.out_file = osp.join(out_root, f'split_{split}_annotated.jsonl')
        
        data_path = f"data/annt/{split}/{split}"
        self.map_root = f"data/maps/gmap_floor1_mpp_0.05_channel_last_with_bounds"
        
        os.makedirs("out", exist_ok=True)

        with gzip.open(data_path+".json.gz", "rt") as f:
            annt_json = json.load(f)
        
        with gzip.open(data_path+"_gt.json.gz", "rt") as f:
            self.gt_json = json.load(f)

        episodes = annt_json['episodes']
        self.sample_eps = random.sample(episodes, TOT_SAMPLES)
        self.cnt = 0
        self.reset_ep()

    def reset_ep(self):
        if self.cnt == TOT_SAMPLES:
            exit()

        episode = self.sample_eps[self.cnt]
        self.ep_id = episode["episode_id"]
        gt_annt = self.gt_json[str(self.ep_id)]

        scene_name = episode['scene_id'].replace('.',' ').split('/')[1]
        instruction = episode['instruction']['instruction_text']

        language_action_seq = instruction.translate(str.maketrans('', '', string.punctuation)).strip().split()
        language_action_seq = list(filter(lambda k: k in direction_list, language_action_seq))

        self.text.setText(f"<b>Instruction of ep_{self.ep_id}_scene_{scene_name}:</b>" +instruction+'<br>'+str(language_action_seq)+"<br>[start from blue to red point]")
        self.text.setAlignment(QtCore.Qt.AlignLeft)
        self.text.setWordWrap(True)

        start_position = episode['start_position']
        end_position = episode['goals'][0]['position']

        nav_map, room_map, obj_maps, grid_dimensions, bounds\
             = get_maps(scene_name, self.map_root, merged=self.merged)

        self.room_map = room_map > 0
        self.obj_maps = obj_maps > 0

        self.nav_map = np.copy(nav_map)
        nav_map = colorize_nav_map(nav_map)

        upper_bound, lower_bound = bounds[0], bounds[1]
        self.grid_dimensions =grid_dimensions
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        # Agent positions
        start_grid_pos = simloc2maploc(
            start_position, grid_dimensions, upper_bound, lower_bound
        )
        self.start_point = start_grid_pos
        end_grid_pos = simloc2maploc(
            end_position, grid_dimensions, upper_bound, lower_bound
        )
        end_radius = episode['goals'][0]['radius']
        # Draw nav map
        draw_agent(start_grid_pos, episode['start_rotation'], nav_map)
        self.map = Image.fromarray(np.copy(nav_map))
        self.update_image(self.map.convert("RGB"))

        draw_path(nav_map, gt_annt, grid_dimensions, upper_bound, lower_bound)

        # Draw answers (ground truth path and end position)
        pil_nav_img = Image.fromarray(nav_map)
        #draw_point(self.pil_nav_img, start_grid_pos[1], start_grid_pos[0], point_size=10, color=(0, 0, 255, 255))
        draw_point(pil_nav_img, end_grid_pos[1], end_grid_pos[0], point_size=10, color=(255, 0, 0, 255))
        overlay = Image.new('RGBA', pil_nav_img.size, (255,0,0)+(0,))
        draw_point(overlay, end_grid_pos[1], end_grid_pos[0], point_size=int(end_radius/0.05), color=(255, 0, 0, 50))

        self.gt_path = Image.alpha_composite(pil_nav_img, overlay)

        # show surrounding objects
        result_dict, relative_dict = get_surrounding_objs(
            obj_maps, self.nav_map, (start_grid_pos[0], start_grid_pos[1], episode['start_rotation']), 
            3, floor_idx=1
        )
        # self.start_instances = draw_instances(result_dict, (start_grid_pos[0], start_grid_pos[1], episode['start_rotation']), self.map.size)
        #self.start_instances = draw_instances(relative_dict, (start_grid_pos[0], start_grid_pos[1], episode['start_rotation']), self.map.size)
        
        # Show room compass
        room_satus = get_room_compass(
            room_map, self.nav_map, (start_grid_pos[0], start_grid_pos[1], episode['start_rotation']), 
            radius=10, num_chunks=8
        )
        self.start_instances = draw_room_instances(room_satus, (start_grid_pos[0], start_grid_pos[1], episode['start_rotation']), self.map.size)

        self.cnt += 1
    
    def show_surround_points(self):
        self.map = Image.alpha_composite(self.map, self.start_instances)
        self.update_image(self.map.convert("RGB"))

    def show_ans(self):
        self.map = Image.alpha_composite(self.map, self.gt_path)
        self.update_image(self.map.convert("RGB"))

    def show_candidate_pathes(self):
        overlay_np = np.zeros((self.nav_map.shape[0], self.nav_map.shape[1], 4), dtype=np.uint8)
        targets, pathes = get_possible_paths(self.nav_map>0, self.obj_maps, self.start_point)

        for path in pathes:
            if len(path)==1:
                continue
            draw_path(
                overlay_np, path, self.grid_dimensions, 
                self.upper_bound, self.lower_bound ,is_grid=True, color=(0,200,200,255)
            )

        overlay = Image.fromarray(overlay_np)
        for target in targets:
            draw_point(overlay, target[1], target[0], 3, (255, 200, 200, 255))
        
        self.map = Image.alpha_composite(self.map, overlay)
        self.update_image(self.map.convert("RGB"))

    def setup_ui(self):
        self.map_region = QVBoxLayout()
        self.image_label = QLabel()
        # self.image_label.setPixmap(QPixmap('./frontpage.jpg').scaledToHeight(600))
        # self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.map_region.addWidget(self.image_label)
        
        self.text = QLabel("Dear Annotator, <font color='red'>WELCOME!</font><br>")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse) 
        self.map_region.addWidget(self.text)

        button_valid = QPushButton("Valid instruction")
        button_valid.clicked.connect(lambda: self.annotate(value=True))
        self.map_region.addWidget(button_valid)
        button_invalid = QPushButton("Invalid instruction")
        button_invalid.clicked.connect(lambda: self.annotate(value=False))
        self.map_region.addWidget(button_invalid)

        button_ans = QPushButton("Show answer")
        button_ans.clicked.connect(lambda: self.show_ans())
        self.map_region.addWidget(button_ans)

        button_candidate = QPushButton("Show candidate paths")
        button_candidate.clicked.connect(lambda: self.show_candidate_pathes())
        self.map_region.addWidget(button_candidate)

        button_surround = QPushButton("Show start surrounding")
        button_surround.clicked.connect(lambda: self.show_surround_points())
        self.map_region.addWidget(button_surround)

        button_next = QPushButton("Next episode")
        button_next.clicked.connect(lambda: self.reset_ep())
        self.map_region.addWidget(button_next)

        self.button_layouts = QVBoxLayout()
        self.room_layout = QGridLayout()
        room_label = QLabel("<b>Room Map buttons</b>")
        self.room_layout.addWidget(room_label, 0, 0, 1, 6, QtCore.Qt.AlignLeft)
        self.obj_layout = QGridLayout()
        obj_label = QLabel("<b>Objects Map buttons</b>")
        self.obj_layout.addWidget(obj_label, 0, 0, 1, 8, QtCore.Qt.AlignLeft)

        # room button
        for i in range(6):
            for j in range(5):
                idx = i*5+j
                try:
                    button = QPushButton(self.ordered_room_name[idx][1])
                    button.setFixedWidth(120)
                    # NOTE: very important, otherwise the idx will be replace by the last assignment
                    button.clicked.connect(lambda checked, arg=self.ordered_room_name[idx][0]: self.show_room(arg))
                    self.room_layout.addWidget(button, i+1, j)
                except:
                    break

        # object buttons
        for i in range(8):
            for j in range(5):
                idx = i*5+j
                try:
                    button = QPushButton(self.ordered_obj_name[idx][1])
                    button.setFixedWidth(120)
                    button.clicked.connect(lambda checked, arg=self.ordered_obj_name[idx][0]: self.show_objs(arg))
                    self.obj_layout.addWidget(button, i+1, j)
                except:
                    break
        self.button_layouts.addLayout(self.room_layout)
        self.button_layouts.addLayout(self.obj_layout)

        #organize main layout
        self.main_layout = QHBoxLayout(self)  # adding widgets to layot
        self.main_layout.addLayout(self.map_region)
        self.main_layout.addLayout(self.button_layouts)
        self.setLayout(self.main_layout)  # set layot
        self.cnt = 0

    def update_image(self, pil_img):
        qim = ImageQt(pil_img)
        pix = QPixmap.fromImage(qim)
        self.image_label.setPixmap(pix.scaledToHeight(600))
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

    @QtCore.pyqtSlot()
    def show_room(self, idx):
        overlay = Image.fromarray(recolor_room[self.room_map[:,:,idx].astype(int)])
        map = Image.alpha_composite(self.map, overlay).convert("RGB")
        self.update_image(map)

    @QtCore.pyqtSlot()
    def show_objs(self, idx):
        overlay = Image.fromarray(recolor_object[self.obj_maps[:,:,idx].astype(int)])
        map = Image.alpha_composite(self.map, overlay).convert("RGB")
        self.update_image(map)


    def annotate(self, value):
        # append new annotation to csv file
        with jsonlines.open(self.out_file+f'.{value}', mode='a') as writer:
            writer.write({ "ep_id":self.ep_id, "match level": value})


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer(annt_root='./data', split="train", merged=False)
    viewer.show()
    app.exec_()