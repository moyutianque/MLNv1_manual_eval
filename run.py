import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QApplication, QGridLayout, QHBoxLayout
from matplotlib.pyplot import draw
from constants import semantic_sensor_40cat, roomidx2name
import os.path as osp
import copy
import os
import jsonlines
from PIL.ImageQt import ImageQt
import time
import gzip
import json
import random
# random.seed(0)
from utils.map_tools import get_maps, simloc2maploc, draw_agent, draw_point, draw_path
from PIL import Image
import numpy as np
recolor_room = np.array(
            [[255, 255, 255, 0], [255, 128, 0, 200]], dtype=np.uint8
        )
recolor_object = np.array(
            [[255, 255, 255, 0], [255, 128, 0, 200]], dtype=np.uint8
        )
TOT_SAMPLES=20

ordered_room_name = sorted(roomidx2name.items(), key=lambda item: item[1])
ordered_obj_name = sorted(semantic_sensor_40cat.items(), key=lambda item: item[1])
class ImageViewer(QWidget):

    def __init__(self, annt_root, split):
        QWidget.__init__(self)
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

        scene_name = episode['scene_id'].split('/')[1]
        instruction = episode['instruction']['instruction_text']

        self.text.setText("<b>Instruction:</b>" +instruction+"<br>[start from blue to red point]")
        self.text.setAlignment(QtCore.Qt.AlignLeft)
        self.text.setWordWrap(True)

        start_position = episode['start_position']
        end_position = episode['goals'][0]['position']

        nav_map, self.room_map, self.obj_maps, grid_dimensions, bounds = get_maps(scene_name, self.map_root)
        
        upper_bound, lower_bound = bounds[0], bounds[1]
        # Agent positions
        start_grid_pos = simloc2maploc(
            start_position, grid_dimensions, upper_bound, lower_bound
        )
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
        self.pil_nav_img = Image.fromarray(nav_map)
        #draw_point(self.pil_nav_img, start_grid_pos[1], start_grid_pos[0], point_size=10, color=(0, 0, 255, 255))
        draw_point(self.pil_nav_img, end_grid_pos[1], end_grid_pos[0], point_size=10, color=(255, 0, 0, 255))
        overlay = Image.new('RGBA', self.pil_nav_img.size, (255,0,0)+(0,))
        draw_point(overlay, end_grid_pos[1], end_grid_pos[0], point_size=int(end_radius/0.05), color=(255, 0, 0, 50))
        
        self.pil_nav_img = Image.alpha_composite(self.pil_nav_img, overlay)
        #self.update_image(self.pil_nav_img.convert("RGB"))
    
        self.cnt += 1
    
    def show_ans(self):
        self.map=self.pil_nav_img
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
                button = QPushButton(ordered_room_name[idx][1])
                button.setFixedWidth(120)
                # NOTE: very important, otherwise the idx will be replace by the last assignment
                button.clicked.connect(lambda checked, arg=ordered_room_name[idx][0]: self.show_room(arg))
                self.room_layout.addWidget(button, i+1, j)

        # object buttons
        for i in range(8):
            for j in range(5):
                idx = i*5+j
                button = QPushButton(ordered_obj_name[idx][1])
                button.setFixedWidth(120)
                button.clicked.connect(lambda checked, arg=ordered_obj_name[idx][0]: self.show_objs(arg))
                self.obj_layout.addWidget(button, i+1, j)

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
    viewer = ImageViewer(annt_root='./data', split="train")
    viewer.show()
    app.exec_()