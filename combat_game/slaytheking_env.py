import random
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# actions

ATTACK = 0
DEFEND = 1
RECON = 2
HEAL = 3
BOOST = 4

ACTION_NAMES = {
    ATTACK: "attack",
    DEFEND: "defend",
    RECON: "recon",
    HEAL: "heal",
    BOOST: "boost",
}

@dataclass
class Statetus:
    round_idx: int = 1
    player_hp: int = 100
    enemy_hp: int = 100
    player_defend: bool = False
    enemy_defend: bool = False
    attack_boost: bool = False
    pre_round_action: int = 0 # 0: none, 1: attack, 2: defend
    recon_used: bool = False
    done: bool = False
    winner: int = 0 # 0: none, 1: player, 2: enemy

class SlayTheKingEnv(gym.Env):

    metadata = {"render_modes": [None, "ansi"], "render_fps": 4}

    def __init__(self, max_rounds: int = 5, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.max_rounds = max_rounds
        self.render_mode = render_mode
        self.rng = random.Random(seed)

        # Obwervation:
        # [player_hp, enemy_hp, round_norm, player_def, enemy_def,
        #  next_boost, scan_used, extra_skill_unlocked,
        #  revealed_none, revealed_attack, revealed_defend]
        low = np.zeros(11, dtype=np.float32)
        high = np.ones(11, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(5)
        self.state = Statetus()
        self.last_info: Dict = {}

    def seed(self, seed: Optional[int] = None):
        self.rng.seed(seed)

    def _weighted_action(self, items: Tuple[Tuple[str, float], ...]) -> str:
        total = sum(weight for _, weight in items)
        r = self.rng.random() * total
        for value, weight in items:
            if r < weight:
                return value
            r -= weight
        return items[-1][0]

    # TODO: make this playable by MD and MF.
    def _action_by_enemy_hp(self, hp: float) -> str: 
        if hp > 70:
            return self._weighted_action((("attack", 0.75), ("defend", 0.25)))
        if hp > 35:
            return self._weighted_action((("attack", 0.55), ("defend", 0.45)))
        return self._weighted_action((("attack", 0.30), ("defend", 0.70)))

    # Recon, Heal, boost after round 2
    def _extra_skills(self) -> bool:
        return self.state.round_idx >= 3


    def _intent_to_onehot(self, intent_code: int) -> np.ndarray:
        out = np.zeros(3, dtype=np.float32)
        out[intent_code] = 1.0
        return out

    def _get_obs(self) -> np.ndarray:
        s = self.state
        revealed = self._intent_to_onehot(s.revealed_intent)
        obs = np.array([
            s.player_hp / 100.0,
            s.enemy_hp / 100.0,
            s.round_idx / float(self.max_rounds),
            float(s.player_defending),
            float(s.enemy_defending),
            float(s.next_attack_boost_armed),
            float(s.recon_used_this_turn),
            float(self._extra_skills()),
            revealed[0],
            revealed[1],
            revealed[2],
        ], dtype=np.float32)
        return obs

    def _get_valid_action_mask(self) -> np.ndarray:
        mask = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        if self._extra_skills():
            mask[:] = 1
        if self.state.recon_used_this_turn:
            mask[RECON] = 0
        return mask

    def _reset_turn_flags_for_new_player_turn(self):
        self.state.player_defending = False
        self.state.enemy_defending = False
        self.state.revealed_intent = 0
        self.state.recon_used_this_turn = False

    def _start_next_round(self):
        self.state.round_idx += 1
        self.state.enemy_hp = 100.0
        self._reset_turn_flags_for_new_player_turn()

    def _finish_game(self, winner: int):
        self.state.done = True
        self.state.winner = winner

    # TODO: what should the reward be? hp? or reward by actions?
    def _rewards(self, prev_player_hp: float, prev_enemy_hp: float, invalid: bool, won_round: bool, won_game: bool, lost_game: bool) -> float:
        # Dense shaping + sparse terminal bonuses
        reward = 0.0
        reward += (prev_enemy_hp - self.state.enemy_hp) * 0.10
        reward -= (prev_player_hp - self.state.player_hp) * 0.10

        if invalid:
            reward -= 0.20
        if won_round:
            reward += 2.0
        if won_game:
            reward += 10.0
        if lost_game:
            reward -= 10.0
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        self.state = Statetus()
        self.last_info = {
            "round_cleared": False,
            "invalid_action": False,
            "player_action": None,
            "enemy_intent": None,
            "action_mask": self._get_valid_action_mask(),
        }
        return self._get_obs(), self.last_info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action index: {action}"
        if self.state.done:
            raise RuntimeError("step() called after episode done. Call reset() first.")

        prev_player_hp = self.state.player_hp
        prev_enemy_hp = self.state.enemy_hp
        invalid_action = False
        round_cleared = False
        enemy_intent_name = None
        player_action_name = ACTION_NAMES[action]

        valid_mask = self._get_valid_action_mask()
        if valid_mask[action] == 0:
            invalid_action = True
        else:
            # ---- Player turn ----
            if action == ATTACK:
                result = self._weighted_pick((("normal", 0.50), ("power", 0.30), ("self", 0.20)))
                damage = 0.0

                if result == "normal":
                    damage = 10.0
                elif result == "power":
                    damage = 25.0
                else:
                    self.state.player_hp = max(0.0, self.state.player_hp - 10.0)

                if self.state.next_attack_boost_armed:
                    if damage > 0.0:
                        damage = 25.0
                    self.state.next_attack_boost_armed = False

                if self.state.enemy_defending and damage > 0.0:
                    damage = round(damage * 0.5)

                if damage > 0.0:
                    self.state.enemy_hp = max(0.0, self.state.enemy_hp - damage)

            elif action == DEFEND:
                self.state.player_defending = True

            elif action == RECON:
                self.state.player_hp = max(0.0, self.state.player_hp - 5.0)
                predicted = self._get_enemy_intent_by_hp(self.state.enemy_hp)
                self.state.revealed_intent = 1 if predicted == "attack" else 2
                self.state.recon_used_this_turn = True

            elif action == HEAL:
                self.state.player_hp = min(100.0, self.state.player_hp + 30.0)
                # turn ends immediately; no additional player action this step

            elif action == BOOST:
                result = self._weighted_pick((("boost", 0.50), ("penalty", 0.50)))
                if result == "boost":
                    self.state.next_attack_boost_armed = True
                else:
                    self.state.player_hp = max(0.0, self.state.player_hp - 30.0)

        # if player died from self-cost before enemy turn
        if self.state.player_hp <= 0.0:
            self._finish_game(winner=2)
        # if enemy died from player action
        elif self.state.enemy_hp <= 0.0:
            round_cleared = True
            if self.state.round_idx >= self.max_rounds:
                self._finish_game(winner=1)
            else:
                self._start_next_round()

        # ---- Enemy turn ----
        elif not invalid_action:
            if self.state.revealed_intent == 1:
                enemy_intent_name = "attack"
            elif self.state.revealed_intent == 2:
                enemy_intent_name = "defend"
            else:
                enemy_intent_name = self._get_enemy_intent_by_hp(self.state.enemy_hp)

            if enemy_intent_name == "defend":
                self.state.enemy_defending = True
            else:
                damage = 18.0 if self.state.enemy_hp > 50.0 else 14.0
                if self.state.player_defending:
                    damage = round(damage * 0.5)
                self.state.player_hp = max(0.0, self.state.player_hp - damage)

            if self.state.player_hp <= 0.0:
                self._finish_game(winner=2)

        won_game = self.state.done and self.state.winner == 1
        lost_game = self.state.done and self.state.winner == 2
        reward = self._rewards(
            prev_player_hp=prev_player_hp,
            prev_enemy_hp=prev_enemy_hp,
            invalid=invalid_action,
            won_round=round_cleared,
            won_game=won_game,
            lost_game=lost_game,
        )

        if not self.state.done and not round_cleared:
            self._reset_turn_flags_for_new_player_turn()

        self.last_info = {
            "round_cleared": round_cleared,
            "invalid_action": invalid_action,
            "player_action": player_action_name,
            "enemy_intent": enemy_intent_name,
            "winner": self.state.winner,
            "action_mask": self._get_valid_action_mask(),
            "state_dict": asdict(self.state),
        }

        terminated = self.state.done
        truncated = False
        obs = self._get_obs()

        if self.render_mode == "ansi":
            self.render()

        return obs, reward, terminated, truncated, self.last_info

    def render(self):
        s = self.state
        lines = [
            f"Round: {s.round_idx}/{self.max_rounds}",
            f"Player HP: {s.player_hp:.1f}",
            f"Enemy HP: {s.enemy_hp:.1f}",
            f"Player defending: {s.player_defending}",
            f"Enemy defending: {s.enemy_defending}",
            f"Next attack boost armed: {s.next_attack_boost_armed}",
            f"Revealed intent: {s.revealed_intent}",
            f"Done: {s.done}, Winner: {s.winner}",
        ]
        text = "\n".join(lines)
        if self.render_mode == "ansi":
            print(text)
        return text

    def close(self):
        pass


if __name__ == "__main__":
    env = Statetus(render_mode="ansi", seed=125)
    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Initial info:", info)

    done = False
    total_reward = 0.0
    step_idx = 0

    while not done:
        step_idx += 1
        mask = info["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        action = int(random.choice(valid_actions))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"\nStep {step_idx}")
        print("Action:", ACTION_NAMES[action])
        print("Reward:", reward)
        print("Info:", {k: v for k, v in info.items() if k != "state_dict"})

        done = terminated or truncated

    print("\nEpisode finished. Total reward:", total_reward)
