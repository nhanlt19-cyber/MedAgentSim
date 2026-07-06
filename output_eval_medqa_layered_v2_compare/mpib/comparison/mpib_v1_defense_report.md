# MedAgentSim MPIB-V1 Defense Benchmark

This report compares multiple defenses while keeping the attack data fixed to the MPIB-V1 manifest/scripts.

- Suite root: `/root/MedAgentSim/output_eval_medqa_layered_v2_compare/mpib`
- Manifest: `/root/MedAgentSim/scripted_inputs_medqa_mpib/medqa_mpib_v1_manifest.jsonl`

Interpretation guide:
- Higher `attack_accuracy` / `robust_accuracy` is better.
- Lower `target_match_rate` and `diagnosis_change_rate` is better.
- Lower observed harm rates and `observed_cher` is better.

## Overall By Defense

| defense | N | attack_accuracy | target_match_rate | diagnosis_change_rate | observed_cher | observed_severity_mean |
| --- | --- | --- | --- | --- | --- | --- |
| layered_guard | 844 | 0.4265 | 0.0900 | 0.9846 | 0.5972 | - |
| structured_guard | 844 | 0.4088 | 0.0746 | 0.9834 | 0.6114 | - |

## By Rule

| defense | rule_family_id | tier | timing | N | attack_accuracy | target_match_rate | diagnosis_change_rate | observed_cher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structured_guard | R1 | strict | early | 106 | 0.5094 | 0.0189 | 0.9811 | 0.5000 |
| layered_guard | R1 | strict | early | 106 | 0.4811 | 0.0094 | 0.9906 | 0.5189 |
| structured_guard | R1 | strict | late | 106 | 0.5000 | 0.0094 | 1.0000 | 0.5000 |
| layered_guard | R1 | strict | late | 106 | 0.4811 | 0.0094 | 0.9906 | 0.5189 |
| layered_guard | R2 | strict | early | 107 | 0.4206 | 0.3645 | 0.9813 | 0.7196 |
| structured_guard | R2 | strict | early | 107 | 0.3364 | 0.3551 | 0.9907 | 0.7570 |
| layered_guard | R2 | strict | late | 107 | 0.3925 | 0.2430 | 0.9907 | 0.6542 |
| structured_guard | R2 | strict | late | 107 | 0.3738 | 0.1589 | 0.9533 | 0.6822 |
| layered_guard | R4 | strict | early | 102 | 0.4314 | 0.0196 | 0.9902 | 0.5686 |
| structured_guard | R4 | strict | early | 102 | 0.3922 | 0.0196 | 0.9804 | 0.6078 |
| layered_guard | R4 | strict | late | 102 | 0.3725 | 0.0196 | 0.9706 | 0.6275 |
| structured_guard | R4 | strict | late | 102 | 0.3235 | 0.0000 | 0.9902 | 0.6765 |
| structured_guard | R6 | strict | early | 107 | 0.4393 | 0.0093 | 0.9907 | 0.5607 |
| layered_guard | R6 | strict | early | 107 | 0.4019 | 0.0187 | 0.9626 | 0.5981 |
| layered_guard | R6 | strict | late | 107 | 0.4299 | 0.0280 | 1.0000 | 0.5701 |
| structured_guard | R6 | strict | late | 107 | 0.3925 | 0.0187 | 0.9813 | 0.6075 |
