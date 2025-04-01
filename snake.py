import pygame
import random
import numpy as np
from neural_network import NeuralNetwork
from constants import *

class Snake:
    def __init__(self, brain=None, color=GREEN):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.grow = False
        self.alive = True
        self.score = 0
        self.steps_taken = 0 # Số bước đã đi
        self.steps_since_food = 0 # Số bước kể từ lần ăn cuối
        self.color = color

        # Bộ não AI
        if brain:
            self.brain = brain.clone() # Mỗi con rắn có bản sao não riêng
        else:
            # Nếu không có não được cung cấp, tạo não ngẫu nhiên
            self.brain = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)

        # Thuộc tính cho GA
        self.fitness = 0

    def get_head_position(self):
        return self.positions[0]

    def turn(self, new_direction_index, current_direction):
        # new_direction_index: 0=Rẽ trái, 1=Đi thẳng, 2=Rẽ phải (so với hướng hiện tại)
        # current_direction: Hướng hiện tại của rắn (UP, DOWN, LEFT, RIGHT)

        directions_list = [UP, RIGHT, DOWN, LEFT] # Theo chiều kim đồng hồ
        current_index = directions_list.index(current_direction)

        if new_direction_index == 0: # Rẽ trái
            new_idx = (current_index - 1) % 4
        elif new_direction_index == 2: # Rẽ phải
            new_idx = (current_index + 1) % 4
        else: # Đi thẳng (index = 1)
            new_idx = current_index

        new_dir = directions_list[new_idx]

        # Ngăn rắn quay đầu 180 độ
        if (new_dir[0] * -1, new_dir[1] * -1) != current_direction:
             self.direction = new_dir


    def move(self, food_pos, all_snake_bodies):
        if not self.alive:
            return

        self.steps_taken += 1
        self.steps_since_food += 1

        # --- Phần AI quyết định hướng đi ---
        inputs = self.get_inputs(food_pos, all_snake_bodies)
        outputs = self.brain.feedforward(inputs)
        decision = np.argmax(outputs) # Chọn hành động có output cao nhất
        self.turn(decision, self.direction)
        # --- Kết thúc phần AI ---

        cur_x, cur_y = self.get_head_position()
        dir_x, dir_y = self.direction
        new_head = ((cur_x + dir_x), (cur_y + dir_y)) # Tính toán vị trí mới

        # Kiểm tra va chạm tường (ví dụ đơn giản, không vòng qua)
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            self.alive = False
            return

        # Kiểm tra va chạm thân
        if new_head in self.positions[1:]: # Không kiểm tra đầu
            self.alive = False
            return

        # Di chuyển
        self.positions.insert(0, new_head)

        # Xử lý ăn mồi
        if new_head == food_pos:
            self.score += 1
            self.grow = True
            self.steps_since_food = 0 # Reset bộ đếm
            # Có thể thêm phần thưởng lớn cho fitness ở đây
        else:
            # Nếu không lớn lên, bỏ đuôi
            if not self.grow:
                self.positions.pop()
            else:
                self.grow = False

        # Giới hạn số bước không ăn được mồi để tránh vòng lặp vô hạn
        if self.steps_since_food > GRID_WIDTH * GRID_HEIGHT * 1.5: # Ngưỡng tùy chỉnh
             self.alive = False


    def get_inputs(self, food_pos, all_snake_bodies):
        """
        Hàm quan trọng: Lấy thông tin môi trường làm input cho NN.
        Cần định nghĩa rõ ràng các input này. Ví dụ:
        1. Nguy hiểm trước mặt? (0 hoặc 1)
        2. Nguy hiểm bên trái? (0 hoặc 1)
        3. Nguy hiểm bên phải? (0 hoặc 1)
        4. Hướng thức ăn (so với đầu rắn): Trái? (0 hoặc 1)
        5. Hướng thức ăn (so với đầu rắn): Phải? (0 hoặc 1)
        6. Hướng thức ăn (so với đầu rắn): Trước mặt? (0 hoặc 1)
        7. Hướng thức ăn (so với đầu rắn): Phía sau? (0 hoặc 1)
        8. Có thể thêm: Hướng đuôi? Hoặc khoảng cách tới tường/thân theo các hướng.
        """
        inputs = [0] * INPUT_NODES # Khởi tạo mảng input
        head_x, head_y = self.get_head_position()
        food_x, food_y = food_pos

        # Xác định hướng tương đối: Trái, Thẳng, Phải
        # Lấy hướng hiện tại
        current_dir = self.direction
        # Xác định hướng bên trái và phải của hướng hiện tại
        if current_dir == UP:
            dir_l, dir_s, dir_r = LEFT, UP, RIGHT
        elif current_dir == DOWN:
            dir_l, dir_s, dir_r = RIGHT, DOWN, LEFT
        elif current_dir == LEFT:
            dir_l, dir_s, dir_r = DOWN, LEFT, UP
        else: # RIGHT
            dir_l, dir_s, dir_r = UP, RIGHT, DOWN

        # Kiểm tra nguy hiểm (Tường hoặc thân)
        pos_s = (head_x + dir_s[0], head_y + dir_s[1])
        pos_l = (head_x + dir_l[0], head_y + dir_l[1])
        pos_r = (head_x + dir_r[0], head_y + dir_r[1])

        # Input 0: Nguy hiểm Thẳng
        inputs[0] = 1 if self._is_danger(pos_s, all_snake_bodies) else 0
        # Input 1: Nguy hiểm Trái
        inputs[1] = 1 if self._is_danger(pos_l, all_snake_bodies) else 0
        # Input 2: Nguy hiểm Phải
        inputs[2] = 1 if self._is_danger(pos_r, all_snake_bodies) else 0

        # Input 3-5: Hướng thức ăn (đơn giản hóa)
        # So sánh vector từ đầu rắn đến thức ăn với các hướng tương đối
        vec_food = (food_x - head_x, food_y - head_y)

        # Tính dot product để xem góc tương đối
        dot_s = vec_food[0]*dir_s[0] + vec_food[1]*dir_s[1]
        dot_l = vec_food[0]*dir_l[0] + vec_food[1]*dir_l[1]
        dot_r = vec_food[0]*dir_r[0] + vec_food[1]*dir_r[1]

        inputs[3] = 1 if dot_l > 0 else 0 # Thức ăn nghiêng về bên trái
        inputs[4] = 1 if dot_s > 0 else 0 # Thức ăn nghiêng về phía trước
        inputs[5] = 1 if dot_r > 0 else 0 # Thức ăn nghiêng về bên phải

        # Có thể thêm input khác như khoảng cách tới thức ăn, khoảng cách tới tường...
        # Ví dụ Input 6: Khoảng cách tới tường phía trước (normalized)
        dist_s = self._distance_to_wall(head_x, head_y, dir_s)
        inputs[6] = dist_s / max(GRID_WIDTH, GRID_HEIGHT)

        # Ví dụ Input 7: Khoảng cách tới tường bên trái (normalized)
        dist_l = self._distance_to_wall(head_x, head_y, dir_l)
        inputs[7] = dist_l / max(GRID_WIDTH, GRID_HEIGHT)


        # Đảm bảo trả về đúng số lượng input đã định nghĩa
        return inputs[:INPUT_NODES]

    def _is_danger(self, pos, all_snake_bodies):
        x, y = pos
        # Kiểm tra tường
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return True
        # Kiểm tra va chạm thân (của chính nó hoặc rắn khác nếu có)
        if pos in self.positions: # Chỉ kiểm tra thân của chính nó cho đơn giản
             return True
        # Nâng cao: kiểm tra va chạm với thân của rắn khác
        # if any(pos in snake_body for snake_body in all_snake_bodies):
        #     return True
        return False

    def _distance_to_wall(self, x, y, direction):
        dist = 0
        curr_x, curr_y = x, y
        while True:
            curr_x += direction[0]
            curr_y += direction[1]
            if not (0 <= curr_x < GRID_WIDTH and 0 <= curr_y < GRID_HEIGHT):
                break
            dist += 1
        return dist


    def calculate_fitness(self):
        # Hàm đánh giá độ tốt của con rắn
        # Kết hợp điểm số và thời gian sống
        # Công thức cần tinh chỉnh nhiều!
        # Ví dụ: Ưu tiên điểm cao hơn
        self.fitness = self.steps_taken + (2**self.score) + (self.score**2.1)*500
        # Phạt nếu chết sớm hoặc không ăn được gì
        if self.steps_taken < 10 and self.score == 0:
            self.fitness *= 0.1
        if self.score == 0 and self.steps_since_food > 50: # Phạt nếu loanh quanh lâu mà ko ăn
             self.fitness *= 0.5
        self.fitness = max(1, self.fitness) # Đảm bảo fitness không âm


    def draw(self, surface):
        if not self.alive: return # Không vẽ rắn đã chết

        for i, p in enumerate(self.positions):
            rect = pygame.Rect(p[0] * GRID_SIZE, p[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            inner_rect = pygame.Rect(p[0] * GRID_SIZE + 1, p[1] * GRID_SIZE + 1, GRID_SIZE - 2, GRID_SIZE - 2)
            if i == 0: # Đầu rắn
                 pygame.draw.rect(surface, (0,100,0) , rect) # Màu đậm hơn cho đầu
                 pygame.draw.rect(surface, (0,150,0) , inner_rect)
            else:
                 pygame.draw.rect(surface, self.color, rect)
                 pygame.draw.rect(surface, (self.color[0]*0.8, self.color[1]*0.8, self.color[2]*0.8), inner_rect) # Màu trong nhạt hơn