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

def _dentistry_bay_pitch_and_desk_base():
  """Repeated consultation bays along X in Hospital Dentistry (see arena / object maze)."""
  try:
    pitch = int(os.environ.get("DENTISTRY_BAY_PITCH", "9") or "9")
  except Exception:
    pitch = 9
  try:
    desk_base_x = int(os.environ.get("DENTISTRY_DESK_BASE_X", "52") or "52")
  except Exception:
    desk_base_x = 52
  return pitch, desk_base_x

def _dentistry_bay_k_from_ref_x(ref_x, max_bay=6):
  """Pick bay by closest desk column (more stable than rounding at bay boundaries)."""
  pitch, desk_base_x = _dentistry_bay_pitch_and_desk_base()
  best_k = 0
  best_d = 10**9
  for k in range(max_bay + 1):
    dx = desk_base_x + pitch * k
    d = abs(ref_x - dx)
    if d < best_d:
      best_d = d
      best_k = k
  return best_k

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
    me_arena = ""
    other_arena = ""

  doc_name = "Maria Lopez"
  pat_name = "Klaus Mueller"
  is_clinical_pair = {
    persona.name.strip().lower(),
    other_name.strip().lower(),
  } == {doc_name.lower(), pat_name.lower()}
  target_arena = "dentistry consultation room"
  if not is_clinical_pair:
    return None

  # Clinical demo rule: keep the pair in one exact consultation room.
  # Use the configured seat tiles directly instead of inferring a repeated bay,
  # otherwise the patient may drift into the neighboring room.
  base_doc = _parse_tile_env("DENTISTRY_DOCTOR_SEAT_TILE", (49, 25))
  # Only x=45..53 belongs to this consultation room; x=55 is already the next room.
  # The nearest valid in-room tile on the patient's side is x=52.
  base_pat = _parse_tile_env("DENTISTRY_PATIENT_SEAT_TILE", (52, 25))
  doc_tile = list(base_doc)
  pat_tile = list(base_pat)

  if persona.name.strip().lower() == doc_name.lower():
    return doc_tile
  if persona.name.strip().lower() == pat_name.lower():
    return pat_tile
  return None

def _direct_tile_for_dentistry_room(persona, plan):
  """
  For the clinical demo pair, entering Dentistry should go to fixed seats
  instead of a random walkable tile inside the room.
  """
  if "dentistry consultation room" not in str(plan).strip().lower():
    return None

  name = persona.name.strip().lower()
  if name == "maria lopez":
    return _parse_tile_env("DENTISTRY_DOCTOR_SEAT_TILE", (49, 25))
  if name == "klaus mueller":
    return _parse_tile_env("DENTISTRY_PATIENT_SEAT_TILE", (52, 25))
  return None

def _consultation_path_override(persona, target_tile, maze):
  """
  Keep Klaus on a stable Dentistry route:
  - nudge the early approach one tile farther right to avoid the decorative
    tree line in the first few movement frames, then
  - enter the patient chair from above so he does not cut through the doctor
    or the desk on the final approach.
  """
  name = persona.name.strip().lower()
  patient_seat = _parse_tile_env("DENTISTRY_PATIENT_SEAT_TILE", (52, 25))
  if name != "klaus mueller" or list(target_tile) != list(patient_seat):
    return None

  curr = list(persona.scratch.curr_tile)
  if curr == list(patient_seat):
    return [tuple(curr)]

  # Desired route shape for the demo:
  # 1. Follow the natural corridor until Klaus is clear of the exterior
  #    planter row; do not add an extra right-then-left dogleg.
  # 2. Keep going straight into the room corridor.
  # 3. Turn right near the chair row, then move down into the patient seat.
  room_corridor_tile = [48, 23]
  top_entry_tile = [patient_seat[0], 23]
  try:
    path_to_room_corridor = _cached_path_finder(
      maze.collision_maze,
      curr,
      room_corridor_tile,
      collision_block_id,
    )
    path_to_top_entry = _cached_path_finder(
      maze.collision_maze,
      room_corridor_tile,
      top_entry_tile,
      collision_block_id,
    )
    path_down_to_seat = _cached_path_finder(
      maze.collision_maze,
      top_entry_tile,
      patient_seat,
      collision_block_id,
    )
    if path_to_room_corridor and path_to_top_entry and path_down_to_seat:
      return path_to_room_corridor + path_to_top_entry[1:] + path_down_to_seat[1:]
  except Exception:
    pass

  try:
    direct_path = _cached_path_finder(
      maze.collision_maze,
      curr,
      patient_seat,
      collision_block_id,
    )
    if direct_path and len(direct_path) > 1:
      return direct_path
  except Exception:
    pass
  return None

def _outside_hospital_path_override(persona, target_tile, maze):
  """
  Nudge Klaus away from the decorative tree line outside the hospital before
  letting the normal shortest-path logic continue toward Dentistry.
  """
  name = persona.name.strip().lower()
  if name != "klaus mueller" or not target_tile:
    return None

  patient_seat = _parse_tile_env("DENTISTRY_PATIENT_SEAT_TILE", (52, 25))
  if list(target_tile) == list(patient_seat):
    return None

  curr = list(persona.scratch.curr_tile)
  try:
    curr_det = maze.access_tile(curr) or {}
    curr_arena = str(curr_det.get("arena", "")).strip().lower()
  except Exception:
    curr_arena = ""
  # Only handle the very first outdoor approach. Once Klaus is below this
  # early corridor, let the Dentistry-specific override take over.
  try:
    early_y_floor = int(os.environ.get("OUTSIDE_HOSPITAL_EARLY_Y_FLOOR", "68") or "68")
  except Exception:
    early_y_floor = 68
  if curr[1] < early_y_floor:
    return None

  # Keep the override active for the opening walk-in, even if exact tile
  # metadata briefly flips while Klaus is still hugging the decorative tree
  # line. Shift him a bit farther right than the default route.
  try:
    safe_x_floor = int(os.environ.get("OUTSIDE_HOSPITAL_SAFE_X", "66") or "66")
  except Exception:
    safe_x_floor = 66
  if curr[0] >= safe_x_floor:
    return None
  target_tile = list(target_tile)
  # Force Klaus into a fixed "safe" corridor to the right of the planter line.
  # The previous relative sidestep was still visually too close to the trees.
  safe_x = max(curr[0] + 1, safe_x_floor)
  forced_prefix = [tuple(curr)]
  for step_x in range(curr[0] + 1, safe_x + 1):
    forced_prefix.append((step_x, curr[1]))
  via_tile = [safe_x, curr[1]]

  try:
    path_b = _cached_path_finder(maze.collision_maze, via_tile, target_tile, collision_block_id)
    if path_b and len(path_b) > 1:
      return forced_prefix + path_b[1:]
  except Exception:
    pass

  # Fallback: try a slight diagonal sidestep if the straight one fails.
  for dy in (-1, 1, -2, 2):
    alt_via = [safe_x, curr[1] + dy]
    alt_prefix = [tuple(curr)]
    for step_x in range(curr[0] + 1, safe_x + 1):
      alt_prefix.append((step_x, curr[1]))
    alt_prefix.append((alt_via[0], alt_via[1]))
    try:
      path_b = _cached_path_finder(maze.collision_maze, alt_via, target_tile, collision_block_id)
      if path_b and len(path_b) > 1:
        return alt_prefix + path_b[1:]
    except Exception:
      continue
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
        seat_tile = _direct_tile_for_dentistry_room(persona, base_plan)
        if seat_tile is not None:
          target_tile = seat_tile
        elif base_plan in getattr(maze, "address_tiles", {}):
          target_tile = random.sample(list(maze.address_tiles[base_plan]), 1)[0]
      else:
        # Normal address jump: {world}:{sector}:{arena}:{game_objects}
        seat_tile = _direct_tile_for_dentistry_room(persona, plan)
        if seat_tile is not None:
          target_tile = seat_tile
        elif plan in getattr(maze, "address_tiles", {}):
          target_tile = random.sample(list(maze.address_tiles[plan]), 1)[0]
        else:
          # Sometimes plan can include a game object that isn't indexed; try stripping it.
          base_plan = ":".join(plan.split(":")[:-1])
          seat_tile = _direct_tile_for_dentistry_room(persona, base_plan)
          if seat_tile is not None:
            target_tile = seat_tile
          elif base_plan in getattr(maze, "address_tiles", {}):
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
      seat_tile = _direct_tile_for_dentistry_room(persona, plan)
      if seat_tile is not None:
        target_tiles = [seat_tile]
      else:
        target_tiles = maze.address_tiles[plan]
        target_tiles = random.sample(list(target_tiles), 1)

    else: 
      # This is our default execution. We simply take the persona to the
      # location where the current action is taking place. 
      # Retrieve the target addresses. Again, plan is an action address in its
      # string form. <maze.address_tiles> takes this and returns candidate 
      # coordinates. 
      seat_tile = _direct_tile_for_dentistry_room(persona, plan)
      if seat_tile is not None:
        target_tiles = [seat_tile]
      elif plan not in maze.address_tiles:
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
    if len(target_tiles) == 1:
      override_path = _outside_hospital_path_override(persona, target_tiles[0], maze)
      if not override_path:
        override_path = _consultation_path_override(persona, target_tiles[0], maze)
      if override_path:
        closest_target_tile = target_tiles[0]
        path = override_path
    for i in target_tiles: 
      # path_finder takes a collision_mze and the curr_tile coordinate as 
      # an input, and returns a list of coordinate tuples that becomes the
      # path. 
      # e.g., [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4)...]
      if path is not None:
        break
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

    if path is not None and closest_target_tile is not None:
      outside_override = _outside_hospital_path_override(persona, closest_target_tile, maze)
      if outside_override:
        path = outside_override

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















