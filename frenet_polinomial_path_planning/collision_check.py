'''
refer to
https://blog.csdn.net/weixin_34945803/article/details/106628352?ops_request_misc
=%257B%2522request%255Fid%2522%253A%2522162130407716780269875893%2522%252C%2522scm%2522
%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162130407716780269875893&biz_id
=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-25-106628352.first
_rank_v2_pc_rank_v29&utm_term=%E7%A2%B0%E6%92%9E%E6%A3%80%E6%B5%8B&spm=1018.2226.3001.4187
'''

import matplotlib.pyplot as plt
import numpy as numpy
import math
import time


class Rectangle:
    def __init__(self, center_x, center_y, width, length, heading):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.length = length
        self.heading = heading
        # (x00,y00),(x01,y01)矩形heading方向对称轴线与矩形的两个交点的坐标
        # (x1,y1) -- (x4,y4)为矩形4个顶点坐标
        self.x00 = self.center_x + 0.5 * self.length * math.cos(self.heading)
        self.y00 = self.center_y + 0.5 * self.length * math.sin(self.heading)
        self.x01 = self.center_x - 0.5 * self.length * math.cos(self.heading)
        self.y01 = self.center_y - 0.5 * self.length * math.sin(self.heading)
        self.x1 = self.x00 + 0.5 * self.width * math.cos(self.heading + 0.5 * math.pi)
        self.y1 = self.y00 + 0.5 * self.width * math.sin(self.heading + 0.5 * math.pi)
        self.x2 = self.x00 + 0.5 * self.width * math.cos(self.heading - 0.5 * math.pi)
        self.y2 = self.y00 + 0.5 * self.width * math.sin(self.heading - 0.5 * math.pi)
        self.x3 = self.x01 + (self.x2 - self.x00)
        self.y3 = self.y01 + (self.y2 - self.y00)
        self.x4 = self.x01 + (self.x1 - self.x00)
        self.y4 = self.y01 + (self.y1 - self.y00)
        self.x02 = self.center_x + (self.x2 - self.x00)
        self.y02 = self.center_y + (self.y2 - self.y00)
        self.x03 = self.center_x + (self.x1 - self.x00)
        self.y03 = self.center_y + (self.y1 - self.y00)

    def show_rectangle(self, color='red', line_width=1):
        xs = [self.x1, self.x2, self.x3, self.x4, self.x1]
        ys = [self.y1, self.y2, self.y3, self.y4, self.y1]
        plt.plot(xs, ys, color=color, linewidth=line_width)


def show_rectangles(rectangles, color='red', line_width=1):
    for i in range(len(rectangles)):
        xs = [rectangles[i].x1, rectangles[i].x2, rectangles[i].x3, rectangles[i].x4, rectangles[i].x1]
        ys = [rectangles[i].y1, rectangles[i].y2, rectangles[i].y3, rectangles[i].y4, rectangles[i].y1]
        plt.plot(xs, ys, color=color, linewidth=line_width)


# rectangle1 is ego vehicle, rectangle2 is obstacle or other vehicle
def collision_check_single(rectangle1, rectangle2):
    # ap = math.sqrt((rectangle1.center_x - rectangle2.center_x)**2 + (rectangle1.center_y - rectangle2.center_y)**2)
    if not rectangle2:
        return False
    if rectangle2:
        rectangle1_max_x = max(rectangle1.x1, rectangle1.x2, rectangle1.x3, rectangle1.x4)
        rectangle1_min_x = min(rectangle1.x1, rectangle1.x2, rectangle1.x3, rectangle1.x4)
        rectangle1_max_y = max(rectangle1.y1, rectangle1.y2, rectangle1.y3, rectangle1.y4)
        rectangle1_min_y = min(rectangle1.y1, rectangle1.y2, rectangle1.y3, rectangle1.y4)
        rectangle2_max_x = max(rectangle2.x1, rectangle2.x2, rectangle2.x3, rectangle2.x4)
        rectangle2_min_x = min(rectangle2.x1, rectangle2.x2, rectangle2.x3, rectangle2.x4)
        rectangle2_max_y = max(rectangle2.y1, rectangle2.y2, rectangle2.y3, rectangle2.y4)
        rectangle2_min_y = min(rectangle2.y1, rectangle2.y2, rectangle2.y3, rectangle2.y4)
        if rectangle1_max_x < rectangle2_min_x or rectangle1_min_x > rectangle2_max_x or rectangle1_max_y < rectangle2_min_y or rectangle1_min_y > rectangle2_max_y:
            return False
        else:
            shift_x = rectangle2.center_x - rectangle1.center_x
            shift_y = rectangle2.center_y - rectangle1.center_y

            dx1 = math.cos(rectangle1.heading) * 0.5 * rectangle1.length
            dy1 = math.sin(rectangle1.heading) * 0.5 * rectangle1.length
            dx2 = math.sin(rectangle1.heading) * 0.5 * rectangle1.width
            dy2 = -math.cos(rectangle1.heading) * 0.5 * rectangle1.width

            dx3 = math.cos(rectangle2.heading) * 0.5 * rectangle2.length
            dy3 = math.sin(rectangle2.heading) * 0.5 * rectangle2.length
            dx4 = math.sin(rectangle2.heading) * 0.5 * rectangle2.width
            dy4 = -math.cos(rectangle2.heading) * 0.5 * rectangle2.width

            collision1 = abs(shift_x * math.cos(rectangle1.heading) + shift_y * math.sin(rectangle1.heading)) <= abs(
                dx3 * math.cos(rectangle1.heading) + dy3 * math.sin(rectangle1.heading)) + abs(
                dx4 * math.cos(rectangle1.heading) + dy4 * math.sin(rectangle1.heading)) + 0.5*rectangle1.length
            collision2 = abs(shift_x * math.sin(rectangle1.heading) - shift_y * math.cos(rectangle1.heading)) <= abs(
                dx3 * math.sin(rectangle1.heading) - dy3 * math.cos(rectangle1.heading)) + abs(
                dx4 * math.sin(rectangle1.heading) - dy4 * math.cos(rectangle1.heading)) + 0.5*rectangle1.width
            collision3 = abs(shift_x * math.cos(rectangle2.heading) + shift_y * math.sin(rectangle2.heading)) <= abs(
                dx1 * math.cos(rectangle2.heading) + dy1 * math.sin(rectangle2.heading)) + abs(
                dx2 * math.cos(rectangle2.heading) + dy2 * math.sin(rectangle2.heading)) + 0.5*rectangle2.length
            collision4 = abs(shift_x * math.sin(rectangle2.heading) - shift_y * math.cos(rectangle2.heading)) <= abs(
                dx1 * math.sin(rectangle2.heading) - dy1 * math.cos(rectangle2.heading)) + abs(
                dx2 * math.sin(rectangle1.heading) - dy2 * math.cos(rectangle1.heading)) + 0.5*rectangle2.width
            return collision1 and collision2 and collision3 and collision4


def collision_check_multi(rectangle1, rectangles):
    collision = False
    if not rectangles:
        return False
    else:
        for i in range(len(rectangles)):
            collision = collision_check_single(rectangle1, rectangles[i])
            if collision:
                break
        return collision


def main():
    time1 = time.time()
    rectangle1 = Rectangle(-20, -10, 10, 35, -0.15 * math.pi)
    rectangle2 = Rectangle(10, -10, 16, 20, 0 * math.pi)
    # rectangle3 = Rectangle(10, 10, 12, 15, 0 * math.pi)
    rectangles = [rectangle1, rectangle2]
    collision = collision_check_single(rectangle1, rectangle2)
    time2 = time.time()
    print("collision? ", collision)
    print('cost time', time2-time1)
    rectangle1.show_rectangle('blue', 2)
    rectangle2.show_rectangle("red", 2)
    # plt.plot(x1, y1, x2, y2)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()