from constants import INF


class Node:
    def __init__(self):
        self.g = INF
        self.h = INF
        self.f = INF
        self.is_blocked = False
        self.is_confirmed = False

    def __str__(self):
        return 'g: ' + self.g + ' h: ' + self.h + ' f: ' + self.f + ' is_blocked: ' + str(self.is_blocked)

    def reset(self):
        self.g = INF
        self.h = INF
        self.f = INF
        self.is_blocked = False
        self.is_confirmed = False

    def reset_except_h(self):
        self.g = INF
        self.f = INF
        self.is_blocked = False
        self.is_confirmed = False
