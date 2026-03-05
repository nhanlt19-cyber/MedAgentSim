import os
import subprocess
import threading
import webbrowser
import time
import logging
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Type
from urllib.request import urlopen
from urllib.error import URLError
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use pathlib for better path handling
WORKING_DIR = Path(os.getcwd())

BACKEND_DIR = WORKING_DIR / "Simulacra" / "reverie" / "backend_server"
SIMULATION_CONTROLLER_PATH = BACKEND_DIR / "simulation_controller.json"
CONFIG_PATH = WORKING_DIR / "medsim" / "configs" / "config_sim.yaml"
LOGS_PATH = WORKING_DIR / "logs"

# Create logs directory
LOGS_PATH.mkdir(exist_ok=True)

# Scenario loader classes
from medsim.core.scenario import (
    ScenarioLoaderMedQA,
    ScenarioLoaderMedQAExtended,
    ScenarioLoaderNEJM,
    ScenarioLoaderNEJMExtended,
    ScenarioLoaderMIMICIV,
    resolve_model_name,
)

# Mapping of dataset names to their loader classes
SCENARIO_LOADERS = {
    "MedQA": ScenarioLoaderMedQA,
    "MedQA_Ext": ScenarioLoaderMedQAExtended,
    "NEJM": ScenarioLoaderNEJM,
    "NEJM_Ext": ScenarioLoaderNEJMExtended,
    "MIMICIV": ScenarioLoaderMIMICIV,
}

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and return configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def load_scenario_loader(dataset: str):
    """Get the appropriate scenario loader based on dataset name."""
    loader_class = SCENARIO_LOADERS.get(dataset)
    if not loader_class:
        logger.error(f"Dataset {dataset} does not exist.")
        raise ValueError(f"Dataset {dataset} does not exist.")
    return loader_class()

def print_summary():
    """Print summary of simulation results."""
    try:
        with open(SIMULATION_CONTROLLER_PATH, 'r') as file:
            data = json.load(file)
            
        total_correct = data.get("total_correct", 0)
        total_scenarios = data.get("total_scenarios", 0)
        
        accuracy = (total_correct / total_scenarios) * 100 if total_scenarios > 0 else 0
        
        summary = [
            f"\n===== SIMULATION SUMMARY =====",
            f"Total Correct Diagnoses: {total_correct}",
            f"Total Scenarios Presented: {total_scenarios}",
            f"Overall Accuracy: {accuracy:.2f}%",
            f"==============================="
        ]
        
        for line in summary:
            logger.info(line)
    except Exception as e:
        logger.error(f"Failed to print summary: {e}")

def update_json_file(file_path: Path, updates: Dict[str, Any]):
    """Update JSON file with new values."""
    try:
        # Create default data if file doesn't exist
        if not file_path.exists():
            logger.info(f"File not found. Creating a new file: {file_path}")
            data = {"simulation_active": 0, "simulation_index": 0, "total_scenarios": 0, "total_correct": 0}
        else:
            # Load existing data
            with open(file_path, 'r') as file:
                data = json.load(file)
        
        # Update values
        data.update(updates)
        
        # Save updated data
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            
        return True
    except Exception as e:
        logger.error(f"Failed to update JSON file {file_path}: {e}")
        return False


def _today_date_str() -> str:
    """Return today's date in 'Month D, YYYY' (no leading zero)."""
    now = datetime.now()
    month = calendar.month_name[now.month]
    return f"{month} {now.day}, {now.year}"


def _patch_origin_meta_date(origin_sim_code: str = "test-simulation") -> None:
    """
    Ensure origin simulation meta uses today's date so the UI shows current year.

    This patches:
      Simulacra/environment/frontend_server/storage/<origin>/reverie/meta.json
    - start_date -> today's date
    - curr_time  -> today's date + keep HH:MM:SS
    """
    meta_path = (
        WORKING_DIR
        / "Simulacra"
        / "environment"
        / "frontend_server"
        / "storage"
        / origin_sim_code
        / "reverie"
        / "meta.json"
    )
    if not meta_path.exists():
        logger.warning("Origin meta.json not found at %s", meta_path)
        return

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        today_str = _today_date_str()

        if isinstance(data.get("start_date"), str):
            data["start_date"] = today_str

        curr_time = data.get("curr_time")
        if isinstance(curr_time, str):
            # Expected: "Month D, YYYY, HH:MM:SS"
            try:
                _, time_part = curr_time.rsplit(", ", 1)
                data["curr_time"] = f"{today_str}, {time_part}"
            except ValueError:
                # If unexpected format, still set to midnight.
                data["curr_time"] = f"{today_str}, 00:00:00"
        else:
            data["curr_time"] = f"{today_str}, 00:00:00"

        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Patched origin meta date to today: %s", meta_path)
    except Exception as e:
        logger.warning("Failed patching origin meta.json date (%s): %s", meta_path, e)

def run_backend_server(target: str, stop_event: threading.Event):
    """Run the backend server for a specific target scenario."""
    try:
        # Patch origin meta.json so new scenarios show today's date in UI.
        _patch_origin_meta_date("test-simulation")

        # Backend configuration
        backend_script_file = "reverie.py"
        url = "http://127.0.0.1:8000/simulator_home"
        
        logger.info(f"Running backend server at: {url}")
        logger.info(f"Target scenario: {target}")
        
        # Navigate to backend directory
        os.chdir(BACKEND_DIR)
        logger.info(f"Changed directory to: {BACKEND_DIR}")
        
        # Generate timestamp for log file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = LOGS_PATH / f"{target}_{timestamp}.txt"
        
        # Construct command
        command = f'python "{backend_script_file}" --origin "test-simulation" --target "{target}" --command "toq"'
        logger.info(f"Executing command: {command}")
        
        # Run command with output logging
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            for line in process.stdout:
                logger.info(line.strip())  # Log to console
                log.write(line)  # Write to log file
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Server ran successfully. Logs saved to: {log_file}")
        else:
            logger.error(f"Server failed with return code {process.returncode}. Check logs: {log_file}")
            
    except Exception as e:
        logger.error(f"Error running backend server: {e}")
    finally:
        # Signal completion
        stop_event.set()
        # Change back to original directory
        os.chdir(WORKING_DIR)

def check_frontend_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """Return True if the frontend server is reachable (e.g. http://127.0.0.1:8000)."""
    try:
        urlopen(base_url, timeout=timeout)
        return True
    except (URLError, OSError):
        return False


def check_ollama_reachable(ollama_url: str = "http://localhost:11434", timeout: float = 5.0) -> bool:
    """Return True if Ollama is reachable (backend LLM for Reverie)."""
    try:
        urlopen(f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout)
        return True
    except (URLError, OSError):
        return False


def open_webpage(url: str, delay: int, stop_event: threading.Event):
    """Open the simulation webpage after a delay."""
    try:
        logger.info(f"Waiting {delay} seconds before opening webpage: {url}")
        time.sleep(delay)
        base_url = url.rsplit("/", 1)[0] + "/"
        if not check_frontend_reachable(base_url):
            logger.warning(
                "Frontend server is not reachable at %s. "
                "Start it first in another terminal: python -m medsim.server",
                base_url,
            )
        if not stop_event.is_set():
            logger.info(f"Opening webpage: {url}")
            webbrowser.open(url)
        else:
            logger.info("Webpage opening skipped as backend has finished.")
    except Exception as e:
        logger.error(f"Error opening webpage: {e}")

def run_scenarios(num_scenarios: int, delay: int = 5):
    """Run multiple clinical scenarios in sequence."""
    try:
        if not check_frontend_reachable("http://127.0.0.1:8000/"):
            logger.warning(
                "Frontend not reachable at http://127.0.0.1:8000/. "
                "Start it first: python -m medsim.server (keep it running)."
            )
        if not check_ollama_reachable():
            logger.warning(
                "Ollama is not reachable at http://localhost:11434. "
                "The simulation will hang when Reverie calls the LLM. Start Ollama first "
                "(e.g. run 'ollama serve' in a terminal or systemctl start ollama), "
                "then run this again. Check with: curl http://localhost:11434/api/tags"
            )
        # Initialize counters
        total_scenarios = 0
        total_correct = 0
        
        # Reset simulation state
        update_json_file(
            SIMULATION_CONTROLLER_PATH, 
            {
                "total_scenarios": total_scenarios, 
                "total_correct": total_correct, 
                "num_scenarios": num_scenarios
            }
        )
        
        # Run each scenario
        for i in range(num_scenarios):
            logger.info(f"\n=== Starting Scenario {i+1}/{num_scenarios} ===")
            
            # Update scenario index and reset diagnosis flag so Reverie can set it when DIAGNOSIS READY
            update_json_file(SIMULATION_CONTROLLER_PATH, {"simulation_index": i, "diagnosis_ready": False})
            
            # Setup for this scenario
            target = f"scenario-{i}"
            url = "http://127.0.0.1:8000/simulator_home"
            stop_event = threading.Event()
            
            # Create and start threads
            backend_thread = threading.Thread(
                target=run_backend_server, 
                args=(target, stop_event),
                name=f"Backend-{i}"
            )
            
            webpage_thread = threading.Thread(
                target=open_webpage, 
                args=(url, delay, stop_event),
                name=f"Webpage-{i}"
            )
            
            backend_thread.start()
            webpage_thread.start()
            
            # Wait for completion
            backend_thread.join()
            webpage_thread.join()
            
            logger.info(f"=== Scenario {i+1}/{num_scenarios} completed ===\n")
            
        logger.info("All scenarios have completed.")
        print_summary()
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
        print_summary()
    except Exception as e:
        logger.error(f"Error running scenarios: {e}")
        print_summary()

def parse_args():
    p = argparse.ArgumentParser(description="Run MedAgentSim clinical scenarios")
    p.add_argument(
        "-n",
        "--num-scenarios",
        type=int,
        help="Number of scenarios to run (overrides config file)",
    )
    return p.parse_args()


def main():
    """Main function to run the simulation."""
    try:
        # parse CLI arguments
        args = parse_args()

        # Load configuration
        logger.info(f"Loading configuration from {CONFIG_PATH}")
        config = load_config(CONFIG_PATH)
        
        # If config specifies a remote LLM server, export it so BAgent picks it up
        if config.get("remote_llm"):
            rl = config["remote_llm"]
            if "url" in rl:
                os.environ.setdefault("SERVER_URL", rl["url"])
            if "token" in rl:
                os.environ.setdefault("SERVER_TOKEN", rl["token"])

        # Initialize scenario loader
        dataset = config["scenario"]["dataset"]
        logger.info(f"Using dataset: {dataset}")
        scenario_loader = load_scenario_loader(dataset)
        
        # Determine number of scenarios
        total_available = scenario_loader.num_scenarios
        if args.num_scenarios is not None:
            num_scenarios = args.num_scenarios
        else:
            configured_num = config["scenario"]["num_scenarios"]
            num_scenarios = configured_num or total_available
        
        logger.info(f"Running {num_scenarios} scenarios (out of {total_available} available)")
        
        # Run the simulation
        # Sử dụng đúng số scenario từ config hoặc CLI
        run_scenarios(num_scenarios)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")

if __name__ == "__main__":
    logger.info("Starting clinical scenario simulation")
    main()