import argparse
import os
import json
import time
import sys
import openai
from medsim.core.agent import MeasurementAgent, PatientAgent, DoctorAgent, compare_results
from medsim.core.scenario import *
from medsim.query_model import *

SCRIPT_END_MARKER = "<<<END>>>"

def _apply_env_model_overrides(doctor_llm, patient_llm, measurement_llm, moderator_llm):
    """
    Allow exact remote model names to override CLI defaults when running
    medsim/main.py directly.
    Priority:
    1. <ROLE>_LLM_MODEL
    2. REMOTE_LLM_MODEL
    """
    shared = (os.environ.get("REMOTE_LLM_MODEL") or "").strip()
    doctor_llm = (os.environ.get("DOCTOR_LLM_MODEL") or "").strip() or shared or doctor_llm
    patient_llm = (os.environ.get("PATIENT_LLM_MODEL") or "").strip() or shared or patient_llm
    measurement_llm = (os.environ.get("MEASUREMENT_LLM_MODEL") or "").strip() or shared or measurement_llm
    moderator_llm = (os.environ.get("MODERATOR_LLM_MODEL") or "").strip() or shared or moderator_llm
    return doctor_llm, patient_llm, measurement_llm, moderator_llm

try:
    # Prefer the shared implementation used by medsim.run / medsim.simulate
    from medsim.run import find_next_available_scenario_id
except Exception:
    def find_next_available_scenario_id(output_dir: str) -> int:
        """
        Fallback: find next available output/scenario_<id> folder index.
        Returns 0 if no scenario folders exist.
        """
        if not os.path.exists(output_dir):
            return 0
        max_id = -1
        for item in os.listdir(output_dir):
            full = os.path.join(output_dir, item)
            if not os.path.isdir(full):
                continue
            suffix = None
            if item.startswith("scenario_"):
                suffix = item.replace("scenario_", "", 1)
            elif item.startswith("scenario-"):
                suffix = item.replace("scenario-", "", 1)
            if suffix is None:
                continue
            try:
                max_id = max(max_id, int(suffix))
            except ValueError:
                continue
        return max_id + 1


def _read_human_input(prompt: str) -> str:
    """
    Reads input for human_doctor / human_patient.

    By default uses Python input() (single-line). Multi-line paste (EscapeChar/Combine attacks)
    does NOT work reliably with input(): only the first line is consumed, the remaining lines
    spill into the next prompt.

    To enable robust multi-line input, set:
      export MULTILINE_HUMAN_INPUT=1
    Then terminate your pasted block with a line containing exactly:
      <<<END>>>
    """
    if os.environ.get("MULTILINE_HUMAN_INPUT", "").strip() not in ("1", "true", "TRUE", "yes", "YES"):
        return input(prompt)

    print(prompt, end="", flush=True)
    print("(multiline enabled: end with a line '<<<END>>>' )", flush=True)
    lines: list[str] = []
    while True:
        line = sys.stdin.readline()
        if line == "":
            break  # EOF
        line = line.rstrip("\n")
        if line.strip() == SCRIPT_END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines).strip()


class ScriptedHumanInput:
    """
    Deterministic replacement for interactive human_patient input.

    File format:
    - JSON object with key "responses": [ ... ]
      Optional keys:
      - "fallback_response": str
      - "repeat_last_on_exhaustion": bool (default: true)
      - "injection_turn": int (0-based index into responses; only for attack scripts) —
        used by main loop to flush patient lines after REQUEST TEST / before final doctor turn
        so injected text is not skipped when the doctor orders labs without another Q&A round.
    - or a root JSON array of strings
    """

    def __init__(self, script_path: str):
        self.script_path = os.path.abspath(script_path)
        with open(self.script_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.injection_turn = None
        if isinstance(payload, dict):
            self.name = payload.get("name") or os.path.basename(self.script_path)
            responses = payload.get("responses")
            self.fallback_response = (payload.get("fallback_response") or "").strip() or None
            self.repeat_last_on_exhaustion = payload.get("repeat_last_on_exhaustion", True)
            atk = payload.get("attack")
            if atk and str(atk).lower() != "none" and isinstance(payload.get("injection_turn"), int):
                self.injection_turn = int(payload["injection_turn"])
        elif isinstance(payload, list):
            self.name = os.path.basename(self.script_path)
            responses = payload
            self.fallback_response = None
            self.repeat_last_on_exhaustion = True
        else:
            raise ValueError(
                f"Unsupported scripted input format in {self.script_path!r}. "
                "Expected a JSON object with 'responses' or a JSON array."
            )

        if not isinstance(responses, list) or not all(isinstance(x, str) for x in responses):
            raise ValueError(
                f"Invalid 'responses' in {self.script_path!r}. Expected a list of strings."
            )

        self.responses = [x.strip() for x in responses if x is not None]
        self.index = 0

    def next_response(self, prompt: str) -> str:
        if self.index >= len(self.responses):
            if self.fallback_response:
                response = self.fallback_response
                mode = "fallback_response"
            elif self.repeat_last_on_exhaustion and self.responses:
                response = self.responses[-1]
                mode = "repeat_last_response"
            else:
                raise RuntimeError(
                    f"Scripted input exhausted for {self.script_path!r} after {self.index} responses. "
                    "Add more responses to the script, set 'fallback_response', or enable "
                    "'repeat_last_on_exhaustion'."
                )

            print(prompt, end="", flush=True)
            print(
                f"(scripted input exhausted after {self.index} responses in {self.name}; using {mode})"
            )
            print(response)
            return response

        response = self.responses[self.index]
        self.index += 1
        print(prompt, end="", flush=True)
        print(f"(scripted input {self.index}/{len(self.responses)} from {self.name})")
        print(response)
        return response


def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm,
         measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences,
         anthropic_api_key=None, output_dir=None, server_url: str | None = None,
         server_token: str | None = None, human_patient_script: str | None = None):
    openai.api_key = api_key
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Remote OpenAI-compatible chat endpoint (vLLM, etc).
    # BAgent/query_model reads these env vars.
    if server_url:
        os.environ["SERVER_URL"] = server_url
    if server_token:
        os.environ["SERVER_TOKEN"] = server_token

    doctor_llm, patient_llm, measurement_llm, moderator_llm = _apply_env_model_overrides(
        doctor_llm, patient_llm, measurement_llm, moderator_llm
    )

    # Default to a local ./output folder for portability (Windows/Linux/macOS).
    output_dir = output_dir or os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    # Load the appropriate scenario loader
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    # Resolve model names from aliases
    doctor_llm = resolve_model_name(doctor_llm)
    patient_llm = resolve_model_name(patient_llm)
    measurement_llm = resolve_model_name(measurement_llm)
    moderator_llm = resolve_model_name(moderator_llm)
    print(
        f"Effective LLMs: Doctor={doctor_llm}, Patient={patient_llm}, "
        f"Measurement={measurement_llm}, Moderator={moderator_llm}"
    )
    total_correct = 0
    total_presents = 0

    if human_patient_script and inf_type != "human_patient":
        raise ValueError("--human_patient_script only works with --inf_type human_patient")

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    if human_patient_script and num_scenarios != 1:
        raise ValueError(
            "--human_patient_script currently supports exactly one scenario per run. "
            "Use --num_scenarios 1 and choose the case with --start_scenario."
        )
    
    meas_agent = MeasurementAgent(backend_str=measurement_llm)
    patient_agent = PatientAgent(backend_str=patient_llm)
    doctor_agent = DoctorAgent(backend_str=doctor_llm)
    mpipe = BAgent(moderator_llm)
    # Cho phép bắt đầu từ một scenario tùy ý (để dễ test từng ca riêng lẻ)
    start_id = globals().get("START_SCENARIO_ID", 0)

    # Auto-increment output scenario id to avoid overwriting output/scenario_0
    start_output_scenario_id = find_next_available_scenario_id(output_dir)

    for loop_idx, _scenario_id in enumerate(
        range(start_id, min(start_id + num_scenarios, scenario_loader.num_scenarios))
    ):
        total_presents += 1
        pi_dialogue = ""
        dialogue_history = []
        scripted_reader = (
            ScriptedHumanInput(human_patient_script) if human_patient_script else None
        )
        
        # Initialize scenario and agents
        scenario = scenario_loader.get_scenario(id=_scenario_id)

        meas_agent.update_scenario(
            scenario=scenario)
        patient_agent.update_scenario(
            scenario=scenario, 
            bias_present=patient_bias)
        doctor_agent.update_scenario(
            scenario=scenario, 
            bias_present=doctor_bias,
            max_infs=total_inferences, 
            img_request=img_request)
        doctor_dialogue = ""        
        for _inf_id in range(total_inferences):
            # Attack scripts: ensure lines up to injection_turn are spoken before the final doctor
            # call. Otherwise "late" inject on the last response is skipped when the doctor
            # requests tests (no patient turn) or finishes early on the last slot.
            if (
                inf_type == "human_patient"
                and scripted_reader is not None
                and getattr(scripted_reader, "injection_turn", None) is not None
                and _inf_id == total_inferences - 1
            ):
                inj = scripted_reader.injection_turn
                while scripted_reader.index <= inj:
                    pi_dialogue = scripted_reader.next_response("\nResponse to doctor: ")
                    patient_text = f"Patient [{int(((_inf_id + 1) / total_inferences) * 100)}%]: {pi_dialogue}"
                    print(patient_text)
                    meas_agent.add_hist(pi_dialogue)
                    dialogue_history.append({"speaker": "Patient", "text": pi_dialogue})
                    time.sleep(1.0)

            # Determine if images are requested
            if dataset == "NEJM":
                imgs = "REQUEST IMAGES" in doctor_dialogue if img_request else True
            else:
                imgs = False

            # Check if final inference
            if _inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

            # Obtain doctor's dialogue
            if inf_type == "human_doctor":
                doctor_dialogue = _read_human_input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)

            # Log and print the doctor's dialogue
            dialogue_text = f"Doctor [{int(((_inf_id+1)/total_inferences)*100)}%]: {doctor_dialogue}"
            print(dialogue_text)

            dialogue_history.append({"speaker": "Doctor", "text": doctor_dialogue})

            # Check for diagnosis.
            # On the last turn, if the model fails to emit "DIAGNOSIS READY", force one more
            # final call that requires the format, and if it still doesn't comply, wrap it.
            if _inf_id == total_inferences - 1 and "DIAGNOSIS READY" not in doctor_dialogue:
                forced_prompt = (
                    "FINAL INSTRUCTION:\n"
                    "You must provide your final answer NOW and you must use exactly this format:\n"
                    "DIAGNOSIS READY: <final diagnosis>\n"
                    "Output ONLY that single line. Do not ask any more questions.\n"
                )
                forced_reply = doctor_agent.inference_doctor(
                    pi_dialogue + "\n" + forced_prompt, image_requested=imgs
                )
                dialogue_text = f"Doctor [forced-final]: {forced_reply}"
                print(dialogue_text)
                dialogue_history.append({"speaker": "Doctor", "text": forced_reply})
                doctor_dialogue = forced_reply

                # If the model still didn't comply, wrap the reply so downstream scoring works.
                if "DIAGNOSIS READY" not in doctor_dialogue:
                    wrapped = doctor_dialogue.strip()
                    if not wrapped:
                        wrapped = "Unknown"
                    doctor_dialogue = f"DIAGNOSIS READY: {wrapped}"
                    dialogue_text = f"Doctor [wrapped-final]: {doctor_dialogue}"
                    print(dialogue_text)
                    dialogue_history.append({"speaker": "Doctor", "text": doctor_dialogue})

            if "DIAGNOSIS READY" in doctor_dialogue:
                if (
                    inf_type == "human_patient"
                    and scripted_reader is not None
                    and getattr(scripted_reader, "injection_turn", None) is not None
                    and scripted_reader.index <= scripted_reader.injection_turn
                ):
                    print(
                        "WARNING: Scripted injection_turn was not reached before DIAGNOSIS READY; "
                        "attack patient line(s) never entered the doctor context for this run.",
                        flush=True,
                    )
                correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), mpipe)
                if correctness:
                    total_correct += 1
                result_text = f"\nCorrect answer: {scenario.diagnosis_information()}"
                scene_text = f"Scene {_scenario_id}, The diagnosis was {'CORRECT' if correctness else 'INCORRECT'} ({int((total_correct/total_presents)*100)}%)"
                print(result_text)
                print(scene_text)
                # Add this scenario's conversation to the master log
                dialogue_history.append(
                    {
                        "DIAGNOSIS_READY_Answer": scenario.diagnosis_information(),
                        "DIAGNOSIS_READY_Simulation": f"Scene {_scenario_id}, The diagnosis was "
                        f"{'CORRECT' if correctness else 'INCORRECT'}, "
                        f"{int((total_correct / total_presents) * 100)}%",
                    }
                )
                break

            # Handle medical exam request
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
                measurement_text = f"Measurement [{int(((_inf_id+1)/total_inferences)*100)}%]: {pi_dialogue}"
                print(measurement_text)
                patient_agent.add_hist(pi_dialogue)
                dialogue_history.append(
                    {"speaker": "Measurement", "text": pi_dialogue}
                )
                if (
                    inf_type == "human_patient"
                    and scripted_reader is not None
                    and getattr(scripted_reader, "injection_turn", None) is not None
                ):
                    inj = scripted_reader.injection_turn
                    while scripted_reader.index <= inj:
                        pi_dialogue = scripted_reader.next_response("\nResponse to doctor: ")
                        patient_text = f"Patient [{int(((_inf_id + 1) / total_inferences) * 100)}%]: {pi_dialogue}"
                        print(patient_text)
                        meas_agent.add_hist(pi_dialogue)
                        dialogue_history.append({"speaker": "Patient", "text": pi_dialogue})
                        time.sleep(1.0)
            else:
                # Obtain patient's response
                if inf_type == "human_patient":
                    if scripted_reader is not None:
                        pi_dialogue = scripted_reader.next_response("\nResponse to doctor: ")
                    else:
                        pi_dialogue = _read_human_input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                patient_text = f"Patient [{int(((_inf_id+1)/total_inferences)*100)}%]: {pi_dialogue}"
                print(patient_text)
                meas_agent.add_hist(pi_dialogue)
                dialogue_history.append({"speaker": "Patient", "text": pi_dialogue})
            # Prevent API timeouts
            time.sleep(1.0)
        
        # Save the dialogue history to a JSON file at the end of each scenario
        output_scenario_id = start_output_scenario_id + loop_idx
        scenario_output_dir = os.path.join(output_dir, f"scenario_{output_scenario_id}")
        os.makedirs(scenario_output_dir, exist_ok=True)
        dialogue_file = os.path.join(scenario_output_dir, "dialogue_history.json")
        with open(dialogue_file, "w") as f:
            json.dump(dialogue_history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='llama3b')
    parser.add_argument('--patient_llm', type=str, default='llama3b')
    parser.add_argument('--measurement_llm', type=str, default='llama3b')
    parser.add_argument('--moderator_llm', type=str, default='llama3b')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--start_scenario', type=int, default=0, required=False, help='Index of first scenario to simulate (0-based)')
    parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
    parser.add_argument('--output_dir', type=str, default=None, required=False, help='Where to save scenario outputs (default: ./output)')
    parser.add_argument(
        '--server_url',
        type=str,
        default=os.environ.get("SERVER_URL"),
        required=False,
        help='Remote OpenAI-compatible chat-completions endpoint (e.g. https://.../v1/chat/completions). Also reads from $SERVER_URL.',
    )
    parser.add_argument(
        '--server_token',
        type=str,
        default=os.environ.get("SERVER_TOKEN"),
        required=False,
        help='Bearer token for remote endpoint (optional). Also reads from $SERVER_TOKEN.',
    )
    parser.add_argument(
        '--human_patient_script',
        type=str,
        default=os.environ.get("HUMAN_PATIENT_SCRIPT"),
        required=False,
        help='Path to a JSON file containing scripted patient responses. Only valid with --inf_type human_patient.',
    )
    
    args = parser.parse_args()

    # Lưu chỉ số scenario bắt đầu vào biến toàn cục để hàm main có thể truy cập
    START_SCENARIO_ID = args.start_scenario

    main(
        args.openai_api_key,
        args.replicate_api_key,
        args.inf_type,
        args.doctor_bias,
        args.patient_bias,
        args.doctor_llm,
        args.patient_llm,
        args.measurement_llm,
        args.moderator_llm,
        args.num_scenarios,
        args.agent_dataset,
        args.doctor_image_request,
        args.total_inferences,
        args.anthropic_api_key,
        args.output_dir,
        args.server_url,
        args.server_token,
        args.human_patient_script,
    )


## terminal running bash
# python medsim/main.py --inf_type llm --doctor_bias None --patient_bias None --doctor_llm meta-llama/Llama-3.3-70B-Instruct --patient_llm meta-llama/Llama-3.3-70B-Instruct --measurement_llm meta-llama/Llama-3.3-70B-Instruct --moderator_llm meta-llama/Llama-3.3-70B-Instruct --agent_dataset MedQA --doctor_image_request False --num_scenarios 10 --total_inferences 20
