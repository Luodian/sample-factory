#!/usr/bin/env python3
"""
Convert sampled frames and actions to training data format.
Creates a parquet file with interleaved images and text (actions).
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import json
import base64
from io import BytesIO
import cv2
import gymnasium as gym
import ale_py

# Register Atari environments
try:
    gym.register_envs(ale_py)
except:
    pass

# Enhanced game-specific descriptions
ENHANCED_GAME_INFO = {
    "atari_alien": {
        "full_name": "Alien",
        "genre": "Maze/Shooter",
        "perspective": "Top-down",
        "player_control": "Spaceship",
        "main_objective": "Navigate mazes, collect eggs, avoid/destroy aliens",
        "initial_state": "Player starts at bottom of maze",
        "key_mechanics": ["8-directional movement", "Shooting", "Egg collection"],
        "hazards": ["Aliens", "Pulsar zones", "Time limits"]
    },
    "atari_amidar": {
        "full_name": "Amidar",
        "genre": "Maze/Strategy",
        "perspective": "Top-down",
        "player_control": "Paint roller character",
        "main_objective": "Paint all segments of rectangular grid while avoiding enemies",
        "initial_state": "Player starts on unpainted grid with roaming enemies",
        "key_mechanics": ["4-directional movement", "Paint trailing", "Jump ability"],
        "hazards": ["Amidars (enemies)", "Dead ends", "Time pressure"]
    },
    "atari_assault": {
        "full_name": "Assault",
        "genre": "Fixed Shooter",
        "perspective": "Side-view",
        "player_control": "Defense turret",
        "main_objective": "Defend mothership from waves of enemy saucers",
        "initial_state": "Defense ship positioned at bottom, enemies approaching from top",
        "key_mechanics": ["Horizontal movement", "Upward shooting", "Wave progression"],
        "hazards": ["Enemy saucers", "Crossfire", "Mothership damage"]
    },
    "atari_asterix": {
        "full_name": "Asterix",
        "genre": "Platformer",
        "perspective": "Side-scrolling",
        "player_control": "Asterix character",
        "main_objective": "Collect items and reach end of level while avoiding enemies",
        "initial_state": "Asterix at start of scrolling level",
        "key_mechanics": ["Running", "Jumping", "Item collection"],
        "hazards": ["Romans", "Obstacles", "Falling"]
    },
    "atari_asteroid": {
        "full_name": "Asteroids",
        "genre": "Space Shooter",
        "perspective": "Top-down",
        "player_control": "Spaceship",
        "main_objective": "Destroy asteroids and UFOs while avoiding collisions",
        "initial_state": "Ship in center, asteroids around edges",
        "key_mechanics": ["Rotation", "Thrust", "Shooting", "Hyperspace"],
        "hazards": ["Asteroids", "UFOs", "Collision with debris"]
    },
    # Additional games for better coverage
    "atari_atlantis": {
        "full_name": "Atlantis",
        "genre": "Fixed Shooter",
        "perspective": "Side-view",
        "player_control": "Gun turrets",
        "main_objective": "Defend Atlantis by shooting down invading vessels",
        "initial_state": "City intact with defensive turrets ready",
        "key_mechanics": ["Turret aiming", "Projectile firing"],
        "hazards": ["Enemy ships", "City destruction"]
    },
    "atari_breakout": {
        "full_name": "Breakout",
        "genre": "Paddle/Ball",
        "perspective": "Side-view",
        "player_control": "Paddle",
        "main_objective": "Break all bricks by bouncing ball with paddle",
        "initial_state": "Paddle at bottom, ball ready, full brick wall above",
        "key_mechanics": ["Paddle movement", "Ball deflection", "Angle control"],
        "hazards": ["Ball loss", "Speed increases"]
    },
    "atari_bankheist": {
        "full_name": "Bank Heist",
        "genre": "Maze/Action",
        "perspective": "Top-down",
        "player_control": "Robber",
        "main_objective": "Rob banks while avoiding police in a maze",
        "initial_state": "Robber in maze with police patrolling",
        "key_mechanics": ["4-directional movement", "Bank robbing", "Dynamite usage"],
        "hazards": ["Police cars", "Dead ends", "Time limits"]
    },
    "atari_battlezone": {
        "full_name": "Battle Zone",
        "genre": "Tank Combat",
        "perspective": "First-person 3D",
        "player_control": "Tank",
        "main_objective": "Tank combat in a 3D wireframe battlefield",
        "initial_state": "Tank positioned on battlefield",
        "key_mechanics": ["Tank movement", "Turret rotation", "Projectile firing"],
        "hazards": ["Enemy tanks", "UFOs", "Obstacles"]
    },
    "atari_beamrider": {
        "full_name": "Beam Rider",
        "genre": "Rail Shooter",
        "perspective": "Third-person",
        "player_control": "Ship",
        "main_objective": "Navigate sectors and destroy enemy formations",
        "initial_state": "Ship on beam grid facing enemy waves",
        "key_mechanics": ["Lateral movement", "Shooting", "Torpedo deployment"],
        "hazards": ["Enemy formations", "Obstacles", "Sector sentinels"]
    },
    "atari_berzerk": {
        "full_name": "Berzerk",
        "genre": "Maze Shooter",
        "perspective": "Top-down",
        "player_control": "Humanoid",
        "main_objective": "Navigate maze rooms, shoot robots, avoid Evil Otto",
        "initial_state": "Player in maze room with hostile robots",
        "key_mechanics": ["8-directional movement", "Shooting", "Room navigation"],
        "hazards": ["Robots", "Evil Otto", "Electrified walls"]
    },
    "atari_bowling": {
        "full_name": "Bowling",
        "genre": "Sports",
        "perspective": "Top-down",
        "player_control": "Bowler",
        "main_objective": "Score strikes and spares in 10-pin bowling",
        "initial_state": "Bowler at lane start with full pin setup",
        "key_mechanics": ["Ball positioning", "Angle control", "Spin application"],
        "hazards": ["Gutter balls", "Splits"]
    },
    "atari_boxing": {
        "full_name": "Boxing",
        "genre": "Fighting/Sports",
        "perspective": "Top-down",
        "player_control": "Boxer",
        "main_objective": "Defeat opponent in boxing match",
        "initial_state": "Boxers facing each other in ring",
        "key_mechanics": ["Movement", "Punching", "Blocking", "Distance management"],
        "hazards": ["Opponent punches", "Ring boundaries"]
    },
    "atari_centipede": {
        "full_name": "Centipede",
        "genre": "Fixed Shooter",
        "perspective": "Top-down",
        "player_control": "Bug blaster",
        "main_objective": "Destroy centipede and other insects",
        "initial_state": "Blaster at bottom, centipede descending through mushroom field",
        "key_mechanics": ["Movement", "Rapid firing", "Mushroom management"],
        "hazards": ["Centipede", "Spiders", "Fleas", "Scorpions"]
    },
    "atari_choppercommand": {
        "full_name": "Chopper Command",
        "genre": "Side-scrolling Shooter",
        "perspective": "Side-view",
        "player_control": "Helicopter",
        "main_objective": "Protect trucks from enemy aircraft",
        "initial_state": "Helicopter above convoy of trucks",
        "key_mechanics": ["Flight control", "Shooting", "Convoy protection"],
        "hazards": ["Enemy jets", "Enemy helicopters", "Truck destruction"]
    },
    "atari_crazyclimber": {
        "full_name": "Crazy Climber",
        "genre": "Climbing",
        "perspective": "Side-view",
        "player_control": "Climber",
        "main_objective": "Climb buildings while avoiding obstacles",
        "initial_state": "Climber at bottom of building",
        "key_mechanics": ["Dual-stick climbing", "Obstacle avoidance", "Window grabbing"],
        "hazards": ["Falling objects", "Closing windows", "Birds", "King Kong"]
    },
    "atari_defender": {
        "full_name": "Defender",
        "genre": "Side-scrolling Shooter",
        "perspective": "Side-view",
        "player_control": "Spaceship",
        "main_objective": "Defend humanoids from alien abduction",
        "initial_state": "Ship patrolling planet surface with humanoids",
        "key_mechanics": ["Horizontal flight", "Shooting", "Humanoid rescue", "Smart bombs"],
        "hazards": ["Landers", "Mutants", "Baiters", "Pods", "Swarmers"]
    },
    "atari_demonattack": {
        "full_name": "Demon Attack",
        "genre": "Fixed Shooter",
        "perspective": "Side-view",
        "player_control": "Laser cannon",
        "main_objective": "Destroy waves of demon birds",
        "initial_state": "Cannon at bottom facing demon waves",
        "key_mechanics": ["Horizontal movement", "Shooting", "Wave progression"],
        "hazards": ["Demon birds", "Splitting enemies", "Boss demons"]
    },
    "atari_doubledunk": {
        "full_name": "Double Dunk",
        "genre": "Sports/Basketball",
        "perspective": "Side-view",
        "player_control": "Basketball player",
        "main_objective": "Score more points than opponent in basketball",
        "initial_state": "Players at center court for tip-off",
        "key_mechanics": ["Dribbling", "Shooting", "Stealing", "Blocking"],
        "hazards": ["Shot clock", "Opponent defense", "Out of bounds"]
    },
    "atari_enduro": {
        "full_name": "Enduro",
        "genre": "Racing",
        "perspective": "Behind-car view",
        "player_control": "Race car",
        "main_objective": "Pass required number of cars each day",
        "initial_state": "Car on track ready to race",
        "key_mechanics": ["Steering", "Acceleration", "Overtaking"],
        "hazards": ["Other cars", "Day/night cycles", "Weather conditions"]
    },
    "atari_fishingderby": {
        "full_name": "Fishing Derby",
        "genre": "Sports/Fishing",
        "perspective": "Side-view",
        "player_control": "Fisherman",
        "main_objective": "Catch more fish than opponent",
        "initial_state": "Fishermen on dock with lines in water",
        "key_mechanics": ["Line casting", "Depth control", "Fish reeling"],
        "hazards": ["Shark stealing fish", "Time limit", "Opponent competition"]
    },
    "atari_freeway": {
        "full_name": "Freeway",
        "genre": "Action",
        "perspective": "Top-down",
        "player_control": "Chicken",
        "main_objective": "Cross highway avoiding traffic",
        "initial_state": "Chicken at bottom of highway",
        "key_mechanics": ["Vertical movement", "Timing", "Speed control"],
        "hazards": ["Cars", "Trucks", "Traffic patterns"]
    },
    "atari_frostbite": {
        "full_name": "Frostbite",
        "genre": "Platform",
        "perspective": "Side-view",
        "player_control": "Eskimo",
        "main_objective": "Build igloos by jumping on ice floes",
        "initial_state": "Eskimo on shore with ice floes moving",
        "key_mechanics": ["Jumping", "Ice floe riding", "Igloo building"],
        "hazards": ["Water", "Temperature timer", "Polar bears", "Birds"]
    },
    "atari_gopher": {
        "full_name": "Gopher",
        "genre": "Action",
        "perspective": "Side-view",
        "player_control": "Farmer",
        "main_objective": "Protect carrots from gophers",
        "initial_state": "Farmer with shovel facing gopher holes",
        "key_mechanics": ["Hole filling", "Gopher bonking", "Carrot protection"],
        "hazards": ["Gophers", "Multiple holes", "Carrot theft"]
    },
    "atari_gravitar": {
        "full_name": "Gravitar",
        "genre": "Space Shooter",
        "perspective": "Side-view",
        "player_control": "Spaceship",
        "main_objective": "Navigate planets destroying bunkers and reactors",
        "initial_state": "Ship in space near planet system",
        "key_mechanics": ["Thrust control", "Rotation", "Shooting", "Tractor beam"],
        "hazards": ["Gravity", "Enemy bunkers", "Reactor cores", "Fuel limits"]
    },
    "atari_hero": {
        "full_name": "H.E.R.O.",
        "genre": "Action/Rescue",
        "perspective": "Side-view",
        "player_control": "Roderick Hero with helicopter backpack",
        "main_objective": "Rescue miners trapped in caves",
        "initial_state": "Hero at cave entrance",
        "key_mechanics": ["Flying", "Laser shooting", "Dynamite placement", "Landing"],
        "hazards": ["Cave walls", "Creatures", "Lava", "Power depletion"]
    },
    "atari_icehockey": {
        "full_name": "Ice Hockey",
        "genre": "Sports/Hockey",
        "perspective": "Top-down",
        "player_control": "Hockey player",
        "main_objective": "Score more goals than opponent",
        "initial_state": "Players at center ice for face-off",
        "key_mechanics": ["Skating", "Puck handling", "Shooting", "Checking"],
        "hazards": ["Opponent defense", "Penalties", "Time limit"]
    },
    "atari_jamesbond": {
        "full_name": "James Bond 007",
        "genre": "Action/Driving",
        "perspective": "Top-down/Side-view",
        "player_control": "James Bond/Car/Boat",
        "main_objective": "Complete missions across multiple vehicle stages",
        "initial_state": "Starting first vehicle mission",
        "key_mechanics": ["Driving", "Shooting", "Jumping", "Mission objectives"],
        "hazards": ["Enemy vehicles", "Obstacles", "Time limits", "Missiles"]
    },
    # Additional 29 games to reach full 57 game coverage
    "atari_kangaroo": {
        "full_name": "Kangaroo",
        "genre": "Platform",
        "perspective": "Side-view",
        "player_control": "Mother kangaroo",
        "main_objective": "Rescue joey while avoiding enemies",
        "initial_state": "Mother kangaroo at bottom of screen",
        "key_mechanics": ["Jumping", "Punching", "Climbing ladders"],
        "hazards": ["Monkeys", "Apples", "Boxing gloves"]
    },
    "atari_kongfumaster": {
        "full_name": "Kung-Fu Master",
        "genre": "Beat 'em up",
        "perspective": "Side-scrolling",
        "player_control": "Martial artist",
        "main_objective": "Fight through floors to rescue girlfriend",
        "initial_state": "Fighter at start of first floor",
        "key_mechanics": ["Punching", "Kicking", "Jumping", "Crouching"],
        "hazards": ["Enemies", "Knife throwers", "Dragons", "Boss fighters"]
    },
    "atari_krull": {
        "full_name": "Krull",
        "genre": "Action/Adventure",
        "perspective": "Top-down/Side-view",
        "player_control": "Prince Colwyn",
        "main_objective": "Rescue princess using the Glaive weapon",
        "initial_state": "Hero starting quest",
        "key_mechanics": ["Movement", "Glaive throwing", "Combat"],
        "hazards": ["Slayers", "Beast army", "Time limits"]
    },
    "atari_montezuma": {
        "full_name": "Montezuma's Revenge",
        "genre": "Platform/Puzzle",
        "perspective": "Side-view",
        "player_control": "Explorer Panama Joe",
        "main_objective": "Navigate pyramid collecting treasures",
        "initial_state": "Explorer at pyramid entrance",
        "key_mechanics": ["Jumping", "Key collection", "Door unlocking", "Rope climbing"],
        "hazards": ["Skulls", "Snakes", "Spiders", "Fire", "Disappearing floors"]
    },
    "atari_mspacman": {
        "full_name": "Ms. Pac-Man",
        "genre": "Maze",
        "perspective": "Top-down",
        "player_control": "Ms. Pac-Man",
        "main_objective": "Eat all dots while avoiding ghosts",
        "initial_state": "Ms. Pac-Man in maze center",
        "key_mechanics": ["4-directional movement", "Dot eating", "Power pellet usage"],
        "hazards": ["Ghosts (Blinky, Pinky, Inky, Sue)", "Time limits"]
    },
    "atari_namethisgame": {
        "full_name": "Name This Game (Octopus)",
        "genre": "Underwater Shooter",
        "perspective": "Side-view",
        "player_control": "Scuba diver",
        "main_objective": "Shoot octopus and sharks while managing oxygen",
        "initial_state": "Diver underwater with oxygen tank",
        "key_mechanics": ["Movement", "Shooting", "Oxygen management"],
        "hazards": ["Octopus", "Sharks", "Oxygen depletion"]
    },
    "atari_phoenix": {
        "full_name": "Phoenix",
        "genre": "Fixed Shooter",
        "perspective": "Bottom-up view",
        "player_control": "Spaceship",
        "main_objective": "Destroy waves of phoenix birds and mothership",
        "initial_state": "Ship at bottom facing bird formations",
        "key_mechanics": ["Horizontal movement", "Shooting", "Shield activation"],
        "hazards": ["Phoenix birds", "Eggs", "Mothership", "Dive attacks"]
    },
    "atari_pitfall": {
        "full_name": "Pitfall!",
        "genre": "Platform/Adventure",
        "perspective": "Side-view",
        "player_control": "Pitfall Harry",
        "main_objective": "Collect treasures in jungle within time limit",
        "initial_state": "Harry in jungle",
        "key_mechanics": ["Running", "Jumping", "Vine swinging", "Ladder climbing"],
        "hazards": ["Crocodiles", "Scorpions", "Rolling logs", "Quicksand", "Pits"]
    },
    "atari_pong": {
        "full_name": "Pong",
        "genre": "Sports/Paddle",
        "perspective": "Top-down",
        "player_control": "Paddle",
        "main_objective": "Score points by getting ball past opponent",
        "initial_state": "Paddles on sides, ball in center",
        "key_mechanics": ["Paddle movement", "Ball deflection"],
        "hazards": ["Missing ball", "Opponent scoring"]
    },
    "atari_privateye": {
        "full_name": "Private Eye",
        "genre": "Adventure/Detective",
        "perspective": "Side-scrolling",
        "player_control": "Detective",
        "main_objective": "Solve cases by finding clues and items",
        "initial_state": "Detective starting case",
        "key_mechanics": ["Driving", "Walking", "Item collection", "Interrogation"],
        "hazards": ["Time limits", "Wrong accusations", "Car crashes"]
    },
    "atari_qbert": {
        "full_name": "Q*bert",
        "genre": "Puzzle/Action",
        "perspective": "Isometric",
        "player_control": "Q*bert",
        "main_objective": "Change all cube colors by hopping",
        "initial_state": "Q*bert at pyramid top",
        "key_mechanics": ["Diagonal hopping", "Color changing", "Disc escape"],
        "hazards": ["Coily snake", "Ugg", "Wrong Way", "Red balls", "Falling off"]
    },
    "atari_riverraid": {
        "full_name": "River Raid",
        "genre": "Vertical Shooter",
        "perspective": "Top-down",
        "player_control": "Fighter jet",
        "main_objective": "Navigate river destroying targets",
        "initial_state": "Jet at river start",
        "key_mechanics": ["Flying", "Shooting", "Fuel management", "Speed control"],
        "hazards": ["Helicopters", "Ships", "Jets", "Bridges", "Fuel depletion"]
    },
    "atari_roadrunner": {
        "full_name": "Road Runner",
        "genre": "Platform/Chase",
        "perspective": "Side-view",
        "player_control": "Road Runner",
        "main_objective": "Escape Wile E. Coyote while collecting seeds",
        "initial_state": "Road Runner on desert road",
        "key_mechanics": ["Running", "Jumping", "Speed bursts", "Seed eating"],
        "hazards": ["Wile E. Coyote", "Trucks", "Cliffs", "Boulders"]
    },
    "atari_robotank": {
        "full_name": "Robot Tank",
        "genre": "Tank Combat",
        "perspective": "First-person",
        "player_control": "Robot tank",
        "main_objective": "Destroy enemy tanks in combat",
        "initial_state": "Tank on battlefield",
        "key_mechanics": ["Tank movement", "Aiming", "Firing", "Radar usage"],
        "hazards": ["Enemy tanks", "System damage", "Low visibility"]
    },
    "atari_seaquest": {
        "full_name": "Seaquest",
        "genre": "Underwater Shooter",
        "perspective": "Side-view",
        "player_control": "Submarine",
        "main_objective": "Rescue divers while fighting sea creatures",
        "initial_state": "Submarine underwater",
        "key_mechanics": ["Movement", "Shooting", "Diver rescue", "Oxygen management"],
        "hazards": ["Sharks", "Enemy subs", "Oxygen depletion"]
    },
    "atari_skiing": {
        "full_name": "Skiing",
        "genre": "Sports/Racing",
        "perspective": "Top-down",
        "player_control": "Skier",
        "main_objective": "Complete slalom or downhill course fastest",
        "initial_state": "Skier at course start",
        "key_mechanics": ["Turning", "Speed control", "Gate navigation"],
        "hazards": ["Trees", "Moguls", "Gates", "Course boundaries"]
    },
    "atari_solaris": {
        "full_name": "Solaris",
        "genre": "Space Combat",
        "perspective": "First-person/Map",
        "player_control": "Starship",
        "main_objective": "Navigate galaxy to find planet Solaris",
        "initial_state": "Ship in space sector",
        "key_mechanics": ["Space combat", "Warp navigation", "Planet scanning"],
        "hazards": ["Zylons", "Cobras", "Pirates", "Fuel depletion"]
    },
    "atari_spaceinvaders": {
        "full_name": "Space Invaders",
        "genre": "Fixed Shooter",
        "perspective": "Bottom-up",
        "player_control": "Laser cannon",
        "main_objective": "Destroy alien invaders before they land",
        "initial_state": "Cannon at bottom, invaders descending",
        "key_mechanics": ["Horizontal movement", "Shooting", "Shield usage"],
        "hazards": ["Invaders", "UFOs", "Invader missiles", "Shield destruction"]
    },
    "atari_stargunner": {
        "full_name": "Stargunner",
        "genre": "Side-scrolling Shooter",
        "perspective": "Side-view",
        "player_control": "Spaceship",
        "main_objective": "Destroy enemy waves in space",
        "initial_state": "Ship ready for combat",
        "key_mechanics": ["Movement", "Shooting", "Power-up collection"],
        "hazards": ["Enemy ships", "Asteroids", "Boss enemies"]
    },
    "atari_surround": {
        "full_name": "Surround",
        "genre": "Snake/Tron",
        "perspective": "Top-down",
        "player_control": "Light trail",
        "main_objective": "Force opponent to crash into walls or trails",
        "initial_state": "Players at opposite positions",
        "key_mechanics": ["Direction control", "Trail creation", "Speed control"],
        "hazards": ["Walls", "Own trail", "Opponent trail"]
    },
    "atari_tennis": {
        "full_name": "Tennis",
        "genre": "Sports",
        "perspective": "Side-view",
        "player_control": "Tennis player",
        "main_objective": "Win tennis match by scoring points",
        "initial_state": "Players on tennis court",
        "key_mechanics": ["Movement", "Swing timing", "Shot placement"],
        "hazards": ["Net", "Out of bounds", "Opponent returns"]
    },
    "atari_timepilot": {
        "full_name": "Time Pilot",
        "genre": "Multidirectional Shooter",
        "perspective": "Top-down",
        "player_control": "Fighter plane",
        "main_objective": "Destroy enemies across time periods",
        "initial_state": "Plane in 1910 era",
        "key_mechanics": ["360-degree movement", "Shooting", "Time travel"],
        "hazards": ["Enemy planes", "Jets", "Helicopters", "Missiles", "UFOs"]
    },
    "atari_tutankham": {
        "full_name": "Tutankham",
        "genre": "Maze/Shooter",
        "perspective": "Top-down",
        "player_control": "Explorer",
        "main_objective": "Navigate tomb collecting treasures",
        "initial_state": "Explorer at tomb entrance",
        "key_mechanics": ["Movement", "Shooting", "Key collection", "Flash usage"],
        "hazards": ["Creatures", "Snakes", "Bats", "Time limits"]
    },
    "atari_upndown": {
        "full_name": "Up'n Down",
        "genre": "Racing/Action",
        "perspective": "Isometric",
        "player_control": "Buggy",
        "main_objective": "Collect flags while racing on hilly track",
        "initial_state": "Buggy at track start",
        "key_mechanics": ["Driving", "Jumping", "Flag collection"],
        "hazards": ["Other cars", "Track edges", "Collision"]
    },
    "atari_venture": {
        "full_name": "Venture",
        "genre": "Dungeon Crawler",
        "perspective": "Top-down",
        "player_control": "Adventurer Winky with bow",
        "main_objective": "Collect treasures from dungeon rooms",
        "initial_state": "Winky in dungeon hallway",
        "key_mechanics": ["Movement", "Shooting", "Room navigation"],
        "hazards": ["Monsters", "Hall Monster", "Wall collision"]
    },
    "atari_videopinball": {
        "full_name": "Video Pinball",
        "genre": "Pinball",
        "perspective": "Top-down",
        "player_control": "Flippers",
        "main_objective": "Score points by hitting targets",
        "initial_state": "Ball ready to launch",
        "key_mechanics": ["Flipper control", "Ball nudging", "Plunger control"],
        "hazards": ["Ball drain", "Tilt penalty"]
    },
    "atari_wizardofwor": {
        "full_name": "Wizard of Wor",
        "genre": "Maze Shooter",
        "perspective": "Top-down",
        "player_control": "Worrior",
        "main_objective": "Clear dungeons of monsters",
        "initial_state": "Worrior in dungeon maze",
        "key_mechanics": ["Movement", "Shooting", "Radar usage"],
        "hazards": ["Burwors", "Garwors", "Thorwors", "Worluk", "Wizard"]
    },
    "atari_yarsrevenge": {
        "full_name": "Yars' Revenge",
        "genre": "Shooter",
        "perspective": "Side-view",
        "player_control": "Yar (insect warrior)",
        "main_objective": "Destroy Qotile behind shield",
        "initial_state": "Yar facing shielded Qotile",
        "key_mechanics": ["Flying", "Nibbling shield", "Cannon formation", "Zorlon firing"],
        "hazards": ["Destroyer missile", "Swirl", "Qotile cannon"]
    },
    "atari_zaxxon": {
        "full_name": "Zaxxon",
        "genre": "Isometric Shooter",
        "perspective": "Isometric 3D",
        "player_control": "Fighter craft",
        "main_objective": "Navigate fortress destroying targets",
        "initial_state": "Craft approaching fortress",
        "key_mechanics": ["3D movement", "Altitude control", "Shooting", "Fuel management"],
        "hazards": ["Walls", "Force fields", "Gun turrets", "Missiles", "Fuel tanks"]
    }
}

def generate_enhanced_description(env_name):
    """
    Generate enhanced description for an Atari environment.
    """
    # Get game-specific information
    game_info = ENHANCED_GAME_INFO.get(env_name, {
        "full_name": env_name.replace("atari_", "").title(),
        "genre": "Arcade",
        "perspective": "2D",
        "player_control": "Character/Vehicle",
        "main_objective": "Complete game objectives",
        "initial_state": "Game start"
    })
    
    # Build description parts
    description_parts = []
    
    # Game identification
    description_parts.append(f"Game: {game_info['full_name']} ({game_info.get('genre', 'Arcade')})")    
    description_parts.append(f"View: {game_info.get('perspective', '2D')} 160×210")
    
    # Initial state
    description_parts.append(f"Initial state: {game_info.get('initial_state', 'Game starting')}")
    
    # Player control
    description_parts.append(f"Player controls: {game_info.get('player_control', 'Character')}")
    
    # Starting action if provided - removed as per request
    # if first_action:
    #     action_name = clean_action_text(first_action)
    #     description_parts.append(f"Starting action: {action_name}")
    
    # Mission
    description_parts.append(f"Mission: {game_info.get('main_objective', 'Complete objectives')}")
    
    # Key mechanics
    if game_info.get('key_mechanics'):
        description_parts.append(f"Mechanics: {', '.join(game_info['key_mechanics'])}")
    
    # Hazards
    if game_info.get('hazards'):
        description_parts.append(f"Hazards: {', '.join(game_info['hazards'])}")
    
    # UI elements (standard for most Atari games)
    description_parts.append("HUD/Score visible at top")
    if env_name in ["atari_alien", "atari_assault"]:
        description_parts.append("Status/Lives visible at bottom")
    
    return ". ".join(description_parts)

def clean_action_text(action_text):
    """
    Clean action text by removing the digit prefix.
    E.g., "8: DOWNRIGHT" -> "DOWNRIGHT"
    """
    if ": " in action_text:
        return action_text.split(": ", 1)[1].strip()
    return action_text.strip()


def load_image_as_base64(image_path):
    """Load an image and convert to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (optional)
            max_size = (256, 256)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_image_as_bytes(image_path):
    """Load an image as bytes."""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def process_episode(episode_dir):
    """
    Process a single episode directory.
    Returns a list of dictionaries with frame/action pairs.
    """
    data_points = []
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(episode_dir, 'frame_*.png')))
    
    for frame_file in frame_files:
        # Extract frame number
        frame_num = os.path.basename(frame_file).replace('frame_', '').replace('.png', '')
        
        # Find corresponding action file
        action_file = os.path.join(episode_dir, f'action_{frame_num}.txt')
        
        if not os.path.exists(action_file):
            continue
        
        # Load action text
        with open(action_file, 'r') as f:
            action_raw = f.read().strip()
        
        # Clean action text (remove digit prefix)
        action_clean = clean_action_text(action_raw)
        
        # Load image as bytes
        image_bytes = load_image_as_bytes(frame_file)
        if image_bytes is None:
            continue
        
        # Create data point
        data_point = {
            'frame_path': frame_file,
            'frame_number': int(frame_num),
            'action': action_clean,
            'action_raw': action_raw,
            'image': image_bytes,
            'episode': os.path.basename(episode_dir)
        }
        
        data_points.append(data_point)
    
    return data_points


def create_interleaved_format(data_points, env_name):
    """
    Create interleaved image-text format for training.
    Returns a list of training examples matching the actual training format.
    """
    training_examples = []
    
    # Group by episode
    episodes = {}
    for dp in data_points:
        ep_name = dp['episode']
        if ep_name not in episodes:
            episodes[ep_name] = []
        episodes[ep_name].append(dp)
    
    # Sort each episode by frame number
    for ep_name in episodes:
        episodes[ep_name].sort(key=lambda x: x['frame_number'])
    
    # Create training examples for each episode
    for ep_name, ep_data in episodes.items():
        # Extract randomness from episode name
        randomness = ep_name.split('_rand')[-1] if '_rand' in ep_name else '0.0'
        
        # Create inputs list with interleaved text and image_gen entries
        inputs = []
        images = []
        
        # Generate enhanced description
        enhanced_desc = generate_enhanced_description(env_name)
        
        # Add environment description - clean dictionary with only needed fields
        inputs.append({
            "type": "text",
            "has_loss": 0,
            "text": enhanced_desc
        })
        
        # Interleave frames and actions
        for i, dp in enumerate(ep_data):
            # Add image
            images.append(dp['image'])
            
            # Add image generation marker - clean dictionary with only needed fields
            inputs.append({
                "type": "image_gen",
                "has_loss": 1,
                "image_index": i  # Keep as regular int, will be preserved properly
            })
            
            # Add action (except for last frame) - clean dictionary with only needed fields
            if i < len(ep_data) - 1:
                inputs.append({
                    "type": "text",
                    "has_loss": 0,
                    "text": dp['action'].lower()
                })
        
        # Create training example
        example = {
            'inputs': inputs,
            'images': images,
            'images_front': images[0:1] if images else [],  # First frame as front image
            'environment': env_name,
            'episode': ep_name,
            'num_frames': len(ep_data),
            'randomness': randomness
        }
        
        training_examples.append(example)
    
    return training_examples


def main():
    parser = argparse.ArgumentParser(description='Convert sampled frames to training data')
    parser.add_argument('--input-dir', required=True,
                        help='Input directory containing sampled frames')
    parser.add_argument('--output-file', required=True,
                        help='Output parquet file path')
    parser.add_argument('--env-filter', default=None,
                        help='Filter for specific environment')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to process')
    parser.add_argument('--format', choices=['simple', 'interleaved'], default='interleaved',
                        help='Output format type')
    
    args = parser.parse_args()
    
    print(f"Converting frames from: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    
    # Find all environment directories
    if args.env_filter:
        env_dirs = glob.glob(os.path.join(args.input_dir, f'*{args.env_filter}*'))
    else:
        env_dirs = glob.glob(os.path.join(args.input_dir, '*'))
    
    env_dirs = [d for d in env_dirs if os.path.isdir(d)]
    
    if not env_dirs:
        print("No environment directories found!")
        return 1
    
    print(f"Found {len(env_dirs)} environment(s)")
    
    all_data = []
    
    # Process each environment
    for env_dir in env_dirs:
        env_name = os.path.basename(env_dir)
        print(f"\nProcessing environment: {env_name}")
        
        # Find episode directories
        episode_dirs = sorted(glob.glob(os.path.join(env_dir, 'episode_*')))
        
        if args.max_episodes:
            episode_dirs = episode_dirs[:args.max_episodes]
        
        print(f"  Found {len(episode_dirs)} episode(s)")
        
        env_data_points = []
        
        # Process each episode
        for ep_dir in episode_dirs:
            ep_name = os.path.basename(ep_dir)
            print(f"    Processing {ep_name}...", end='')
            
            ep_data = process_episode(ep_dir)
            env_data_points.extend(ep_data)
            
            print(f" {len(ep_data)} frames")
        
        if args.format == 'interleaved':
            # Create interleaved format
            training_examples = create_interleaved_format(env_data_points, env_name)
            all_data.extend(training_examples)
        else:
            # Simple format - one row per frame
            for dp in env_data_points:
                all_data.append({
                    'environment': env_name,
                    'episode': dp['episode'],
                    'frame_number': dp['frame_number'],
                    'action': dp['action'],
                    'action_raw': dp['action_raw'],
                    'image': dp['image']
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Post-process inputs to ensure clean format (remove None values from dictionaries)
    if 'inputs' in df.columns and args.format == 'interleaved':
        def clean_input_entry(entry):
            """Remove None values from input dictionaries."""
            if entry['type'] == 'text':
                return {'type': 'text', 'has_loss': entry['has_loss'], 'text': entry['text']}
            else:  # image_gen
                return {'type': 'image_gen', 'has_loss': entry['has_loss'], 'image_index': int(entry['image_index'])}
        
        df['inputs'] = df['inputs'].apply(lambda inputs: [clean_input_entry(inp) for inp in inputs])
    
    print(f"\nTotal data points: {len(df)}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Save to parquet
    df.to_parquet(args.output_file, engine='pyarrow', compression='snappy')
    
    print(f"\nSaved to: {args.output_file}")
    
    # Print sample of the data
    if args.format == 'interleaved':
        print("\nSample training example:")
        if len(df) > 0:
            sample = df.iloc[0]
            print(f"  Environment: {sample['environment']}")
            print(f"  Episode: {sample['episode']}")
            print(f"  Num frames: {sample['num_frames']}")
            print(f"  Randomness: {sample['randomness']}")
            print(f"  Number of inputs: {len(sample['inputs'])}")
            print(f"  Number of images: {len(sample['images'])}")
            
            # Show first few input entries
            print(f"\n  First 5 input entries:")
            for i, inp in enumerate(sample['inputs'][:5]):
                if inp['type'] == 'text':
                    print(f"    [{i}] text (loss={inp['has_loss']}): {inp['text'][:50]}...")
                else:
                    print(f"    [{i}] image_gen (loss={inp['has_loss']}): index={inp['image_index']}")
    else:
        print("\nFirst 5 rows:")
        print(df[['environment', 'episode', 'frame_number', 'action']].head())
    
    return 0


if __name__ == '__main__':
    sys.exit(main())