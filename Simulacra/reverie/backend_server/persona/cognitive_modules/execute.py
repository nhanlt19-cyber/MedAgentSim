"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: execute.py
Description: This defines the "Act" module for generative agents. 
"""
import sys
import random
import os
sys.path.append('../../')

from global_methods import *
from path_finder import *
from utils import *

# Cache path_finder results to avoid recomputing shortest paths repeatedly.
# This preserves slow "walk tile-by-tile" behavior but makes backend faster.
_PATH_CACHE: dict[tuple[tuple[int, int], tuple[int, int]], list] = {}
_PATH_CACHE_MAX = 5000

def _cached_path_finder(collision_maze, start_tile, end_tile, block_id):
  key = (tuple(start_tile), tuple(end_tile))
  cached = _PATH_CACHE.get(key)
  if cached is not None:
    return cached
  path = path_finder(collision_maze, start_tile, end_tile, block_id)
  if len(_PATH_CACHE) >= _PATH_CACHE_MAX:
    # crude eviction: clear all (fast and safe for demo)
    _PATH_CACHE.clear()
  _PATH_CACHE[key] = path
  return path

def _parse_tile_env(key: str, default_xy):
  raw = os.environ.get(key, "").strip()
  if not raw:
    return list(default_xy)
  try:
    x_str, y_str = raw.split(",")
    return [int(x_str.strip()), int(y_str.strip())]
  except Exception:
    return list(default_xy)

def _seat_tile_for_consultation(persona, maze, personas, plan):
  """
  Anchor doctor/patient to stable tiles in Dentistry room while conversing.
  """
  if "<persona>" not in plan:
    return None

  try:
    other_name = plan.split("<persona>")[-1].strip()
  except Exception:
    other_name = ""
  if not other_name or other_name not in personas:
    return None

  try:
    me_det = maze.access_tile(persona.scratch.curr_tile) or {}
    other_det = maze.access_tile(personas[other_name].scratch.curr_tile) or {}
    me_arena = str(me_det.get("arena", "")).strip().lower()
    other_arena = str(other_det.get("arena", "")).strip().lower()
  except Exception:
    return None

  target_arena = "dentistry consultation room"
  if target_arena not in me_arena and target_arena not in other_arena:
    return None

  doc_name = "Maria Lopez"
  pat_name = "Klaus Mueller"
  # Map data: only one "chair" object in Dentistry (32247) at (49,25), west of desk.
  # Desk is (52,25). East side has no second chair object in CSV; use first walkable
  # tile past the desk partition (red chair left / white chair right in the tile art).
  doc_tile = _parse_tile_env("DENTISTRY_DOCTOR_SEAT_TILE", (49, 25))
  pat_tile = _parse_tile_env("DENTISTRY_PATIENT_SEAT_TILE", (55, 25))

  if persona.name.strip().lower() == doc_name.lower():
    return doc_tile
  if persona.name.strip().lower() == pat_name.lower():
    return pat_tile
  return None

def execute(persona, maze, personas, plan): 
  """
  Given a plan (action's string address), we execute the plan (actually 
  outputs the tile coordinate path and the next coordinate for the 
  persona). 

  INPUT:
    persona: Current <Persona> instance.  
    maze: An instance of current <Maze>.
    personas: A dictionary of all personas in the world. 
    plan: This is a string address of the action we need to execute. 
       It comes in the form of "{world}:{sector}:{arena}:{game_objects}". 
       It is important that you access this without doing negative 
       indexing (e.g., [-1]) because the latter address elements may not be 
       present in some cases. 
       e.g., "dolores double studio:double studio:bedroom 1:bed"
    
  OUTPUT: 
    execution
  """
  def _is_truthy_env(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")

  # Fast mode (teleport) for UI demos:
  # - Still moves agents to the correct room (jump to a target tile)
  # - Avoids expensive path_finder() calls and multi-step walking
  #
  # Enable with: FAST_CHAT_TELEPORT=1
  # If you want the old behavior "freeze in place", use FAST_CHAT_DEMO=1
  if _is_truthy_env("FAST_CHAT_DEMO"):
    persona.scratch.planned_path = []
    persona.scratch.act_path_set = True
    ret = persona.scratch.curr_tile
    description = f"{persona.scratch.act_description}"
    description += f" @ {persona.scratch.act_address}"
    return ret, persona.scratch.act_pronunciatio, description

  if _is_truthy_env("FAST_CHAT_TELEPORT"):
    # Resolve a target tile without pathfinding.
    target_tile = None

    try:
      if "<persona>" in plan:
        # Prefer fixed consultation seats in dentistry room for stable visuals.
        seat_tile = _seat_tile_for_consultation(persona, maze, personas, plan)
        if seat_tile is not None:
          target_tile = seat_tile
        else:
          # Fall back to "jump near the other persona".
          other_name = plan.split("<persona>")[-1].strip()
          if other_name in personas and getattr(personas[other_name].scratch, "curr_tile", None) is not None:
            target_tile = personas[other_name].scratch.curr_tile
      elif "<waiting>" in plan:
        parts = plan.split()
        if len(parts) >= 3:
          target_tile = [int(parts[1]), int(parts[2])]
      elif "<random>" in plan:
        base_plan = ":".join(plan.split(":")[:-1])
        if base_plan in getattr(maze, "address_tiles", {}):
          target_tile = random.sample(list(maze.address_tiles[base_plan]), 1)[0]
      else:
        # Normal address jump: {world}:{sector}:{arena}:{game_objects}
        if plan in getattr(maze, "address_tiles", {}):
          target_tile = random.sample(list(maze.address_tiles[plan]), 1)[0]
        else:
          # Sometimes plan can include a game object that isn't indexed; try stripping it.
          base_plan = ":".join(plan.split(":")[:-1])
          if base_plan in getattr(maze, "address_tiles", {}):
            target_tile = random.sample(list(maze.address_tiles[base_plan]), 1)[0]
    except Exception:
      target_tile = None

    if target_tile is not None:
      persona.scratch.planned_path = []
      persona.scratch.act_path_set = True
      persona.scratch.curr_tile = target_tile
      description = f"{persona.scratch.act_description}"
      description += f" @ {persona.scratch.act_address}"
      return target_tile, persona.scratch.act_pronunciatio, description

  if "<random>" in plan and persona.scratch.planned_path == []: 
    persona.scratch.act_path_set = False

  # <act_path_set> is set to True if the path is set for the current action. 
  # It is False otherwise, and means we need to construct a new path. 
  if not persona.scratch.act_path_set: 
    # <target_tiles> is a list of tile coordinates where the persona may go 
    # to execute the current action. The goal is to pick one of them.
    target_tiles = None

    print (plan)

    if "<persona>" in plan: 
      # Executing persona-persona interaction.
      seat_tile = _seat_tile_for_consultation(persona, maze, personas, plan)
      if seat_tile is not None:
        target_tiles = [seat_tile]
      else:
        target_p_tile = (personas[plan.split("<persona>")[-1].strip()]
                         .scratch.curr_tile)
        potential_path = _cached_path_finder(maze.collision_maze, 
                                     persona.scratch.curr_tile, 
                                     target_p_tile, 
                                     collision_block_id)
        if len(potential_path) <= 2: 
          target_tiles = [potential_path[0]]
        else: 
          potential_1 = _cached_path_finder(maze.collision_maze, 
                                  persona.scratch.curr_tile, 
                                  potential_path[int(len(potential_path)/2)], 
                                  collision_block_id)
          potential_2 = _cached_path_finder(maze.collision_maze, 
                                  persona.scratch.curr_tile, 
                                  potential_path[int(len(potential_path)/2)+1], 
                                  collision_block_id)
          if len(potential_1) <= len(potential_2): 
            target_tiles = [potential_path[int(len(potential_path)/2)]]
          else: 
            target_tiles = [potential_path[int(len(potential_path)/2+1)]]
    
    elif "<waiting>" in plan: 
      # Executing interaction where the persona has decided to wait before 
      # executing their action.
      x = int(plan.split()[1])
      y = int(plan.split()[2])
      target_tiles = [[x, y]]

    elif "<random>" in plan: 
      # Executing a random location action.
      plan = ":".join(plan.split(":")[:-1])
      target_tiles = maze.address_tiles[plan]
      target_tiles = random.sample(list(target_tiles), 1)

    else: 
      # This is our default execution. We simply take the persona to the
      # location where the current action is taking place. 
      # Retrieve the target addresses. Again, plan is an action address in its
      # string form. <maze.address_tiles> takes this and returns candidate 
      # coordinates. 
      if plan not in maze.address_tiles:
        # maze.address_tiles["Johnson Park:park:park garden"] #ERRORRRRRRR
        print("################ PLAN ################")
        print(plan)
        print("################ TILES ################")
        print(maze.address_tiles.keys())
        maze.address_tiles["the Ville:Outdoors:Outside Hospital"] #ERRORRRRRRR
      else: 
        target_tiles = maze.address_tiles[plan]

    # There are sometimes more than one tile returned from this (e.g., a tabe
    # may stretch many coordinates). So, we sample a few here. And from that 
    # random sample, we will take the closest ones. 
    if len(target_tiles) < 4: 
      target_tiles = random.sample(list(target_tiles), len(target_tiles))
    else:
      target_tiles = random.sample(list(target_tiles), 4)
    # If possible, we want personas to occupy different tiles when they are 
    # headed to the same location on the maze. It is ok if they end up on the 
    # same time, but we try to lower that probability. 
    # We take care of that overlap here.  
    persona_name_set = set(personas.keys())
    new_target_tiles = []
    for i in target_tiles: 
      curr_event_set = maze.access_tile(i)["events"]
      pass_curr_tile = False
      for j in curr_event_set: 
        if j[0] in persona_name_set: 
          pass_curr_tile = True
      if not pass_curr_tile: 
        new_target_tiles += [i]
    if len(new_target_tiles) == 0: 
      new_target_tiles = target_tiles
    target_tiles = new_target_tiles

    # Now that we've identified the target tile, we find the shortest path to
    # one of the target tiles. 
    curr_tile = persona.scratch.curr_tile
    collision_maze = maze.collision_maze
    closest_target_tile = None
    path = None
    for i in target_tiles: 
      # path_finder takes a collision_mze and the curr_tile coordinate as 
      # an input, and returns a list of coordinate tuples that becomes the
      # path. 
      # e.g., [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4)...]
      curr_path = _cached_path_finder(maze.collision_maze, 
                              curr_tile, 
                              i, 
                              collision_block_id)
      if not closest_target_tile: 
        closest_target_tile = i
        path = curr_path
      elif len(curr_path) < len(path): 
        closest_target_tile = i
        path = curr_path

    # Actually setting the <planned_path> and <act_path_set>. We cut the 
    # first element in the planned_path because it includes the curr_tile. 
    persona.scratch.planned_path = path[1:]
    persona.scratch.act_path_set = True
  
  # Setting up the next immediate step. We stay at our curr_tile if there is
  # no <planned_path> left, but otherwise, we go to the next tile in the path.
  ret = persona.scratch.curr_tile
  if persona.scratch.planned_path: 
    ret = persona.scratch.planned_path[0]
    persona.scratch.planned_path = persona.scratch.planned_path[1:]

  description = f"{persona.scratch.act_description}"
  description += f" @ {persona.scratch.act_address}"

  execution = ret, persona.scratch.act_pronunciatio, description
  return execution















