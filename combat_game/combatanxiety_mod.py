import sys
from typing import Optional

import pygame

from combat_game.combatanxiety_env import Statetus, ATTACK, DEFEND, RECON, HEAL, BOOST

pygame.init()

WIDTH, HEIGHT = 1920, 1080
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CombatAnxiety")
CLOCK = pygame.time.Clock()
FPS = 60

# --- Colors ---
WHITE = (255, 255, 255)

# --- FONT ----
FONT = pygame.font.SysFont("Arial", 20)

# --- Action Labels ---
ACTION_LABELS = {
    ATTACK: "공격",
    DEFEND: "방어",
    RECON: "정찰",
    HEAL: "회복",
    BOOST: "버프",
}

ENEMY_ACTION_LABELS = {
    None: "없음",
    "attack": "공격",
    "defend": "방어",
}

class Button:
    def __init__(self, rect, text, color, text_color=WHITE):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.disabled = False

    def draw(self, surface):
        fill = self.color if not self.disabled else (190, 190, 190) # example rgb
        pygame.draw.rect(surface, fill, self.rect, border_radius=10)
        pygame.draw.rect(surface, fill, self.rect, 1, border_radius=10)
        text = FONT.render(self.text, True, self.text_color if not self.disabled else (95, 95, 95))
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def clicked(self, event):
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
            and not self.disabled
        )
    
@property
def state(self):
    return self.env.state

def add_log(self, message: str):
    self.logs.insert(0, message)
    self.logs = self.logs[:30]

def reset_game(self):
    self.obs, self.info = self.env.reset()
    self.done = False
    self.total_reward = 0.0
    self.logs = ["게임 시작"]
    self.update_buttons()

def update_buttons(self):
        mask = self.info.get("action_mask")
        if mask is None:
            mask = [1, 1, 0, 0, 0]
        for action, btn in self.action_buttons.items():
            btn.disabled = bool(self.done) or int(mask[action]) == 0

