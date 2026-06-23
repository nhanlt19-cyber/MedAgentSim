# MedAgentSim MPIB-V1 Defense Benchmark

This report compares multiple defenses while keeping the attack data fixed to the MPIB-V1 manifest/scripts.

- Suite root: `/root/MedAgentSim/output_eval_medqa_defense_suite/mpib`
- Manifest: `/root/MedAgentSim/scripted_inputs_medqa_mpib/medqa_mpib_v1_manifest.jsonl`

Interpretation guide:
- Higher `attack_accuracy` / `robust_accuracy` is better.
- Lower `target_match_rate` and `diagnosis_change_rate` is better.
- Lower observed harm rates and `observed_cher` is better.

## Overall By Defense

| defense | N | attack_accuracy | target_match_rate | diagnosis_change_rate | observed_cher | observed_severity_mean |
| --- | --- | --- | --- | --- | --- | --- |
| response_based | 844 | 0.4526 | 0.0889 | 0.9822 | 0.5865 | - |
| none | 844 | 0.4526 | 0.0995 | 0.9763 | 0.5841 | - |
| structured_guard | 844 | 0.4443 | 0.0687 | 0.9846 | 0.5829 | - |
| known_answer | 844 | 0.4408 | 0.0983 | 0.9893 | 0.6007 | - |
| layered_guard | 844 | 0.4384 | 0.0723 | 0.9799 | 0.5853 | - |
| prompt_guard | 844 | 0.4325 | 0.0652 | 0.9775 | 0.5877 | - |
| llm_based | 844 | 0.4182 | 0.0806 | 0.9822 | 0.6126 | - |
| ppl-10-4.5 | 844 | 0.2666 | 0.0166 | 0.9502 | 0.7346 | - |

## By Rule

| defense | rule_family_id | tier | timing | N | attack_accuracy | target_match_rate | diagnosis_change_rate | observed_cher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | R1 | strict | early | 106 | 0.4623 | 0.0189 | 0.9717 | 0.5377 |
| prompt_guard | R1 | strict | early | 106 | 0.4623 | 0.0094 | 0.9811 | 0.5377 |
| response_based | R1 | strict | early | 106 | 0.4528 | 0.0094 | 0.9811 | 0.5472 |
| known_answer | R1 | strict | early | 106 | 0.4340 | 0.0189 | 0.9811 | 0.5755 |
| structured_guard | R1 | strict | early | 106 | 0.4340 | 0.0094 | 0.9811 | 0.5660 |
| layered_guard | R1 | strict | early | 106 | 0.4057 | 0.0189 | 0.9717 | 0.5943 |
| llm_based | R1 | strict | early | 106 | 0.3774 | 0.0189 | 0.9811 | 0.6226 |
| ppl-10-4.5 | R1 | strict | early | 106 | 0.2358 | 0.0189 | 0.9340 | 0.7642 |
| structured_guard | R1 | strict | late | 106 | 0.5377 | 0.0000 | 0.9717 | 0.4623 |
| none | R1 | strict | late | 106 | 0.4623 | 0.0094 | 0.9906 | 0.5377 |
| prompt_guard | R1 | strict | late | 106 | 0.4434 | 0.0094 | 0.9811 | 0.5566 |
| layered_guard | R1 | strict | late | 106 | 0.4340 | 0.0189 | 0.9623 | 0.5660 |
| known_answer | R1 | strict | late | 106 | 0.4245 | 0.0189 | 0.9906 | 0.5755 |
| response_based | R1 | strict | late | 106 | 0.3962 | 0.0000 | 0.9906 | 0.6038 |
| llm_based | R1 | strict | late | 106 | 0.3774 | 0.0094 | 0.9717 | 0.6226 |
| ppl-10-4.5 | R1 | strict | late | 106 | 0.2642 | 0.0094 | 0.9623 | 0.7358 |
| response_based | R2 | strict | early | 107 | 0.5140 | 0.3645 | 0.9813 | 0.6542 |
| known_answer | R2 | strict | early | 107 | 0.4953 | 0.3832 | 1.0000 | 0.7103 |
| llm_based | R2 | strict | early | 107 | 0.4860 | 0.2897 | 0.9813 | 0.6449 |
| none | R2 | strict | early | 107 | 0.4766 | 0.3925 | 1.0000 | 0.7103 |
| layered_guard | R2 | strict | early | 107 | 0.4579 | 0.3178 | 0.9720 | 0.6916 |
| prompt_guard | R2 | strict | early | 107 | 0.3925 | 0.2991 | 0.9720 | 0.7290 |
| structured_guard | R2 | strict | early | 107 | 0.3925 | 0.2804 | 1.0000 | 0.7383 |
| ppl-10-4.5 | R2 | strict | early | 107 | 0.2617 | 0.0187 | 0.9626 | 0.7383 |
| none | R2 | strict | late | 107 | 0.4953 | 0.3178 | 0.9813 | 0.6075 |
| response_based | R2 | strict | late | 107 | 0.4860 | 0.2991 | 1.0000 | 0.6542 |
| prompt_guard | R2 | strict | late | 107 | 0.4393 | 0.1308 | 0.9720 | 0.5981 |
| structured_guard | R2 | strict | late | 107 | 0.4393 | 0.2056 | 0.9813 | 0.6449 |
| known_answer | R2 | strict | late | 107 | 0.4206 | 0.3084 | 1.0000 | 0.6916 |
| layered_guard | R2 | strict | late | 107 | 0.4112 | 0.1682 | 1.0000 | 0.6262 |
| llm_based | R2 | strict | late | 107 | 0.4112 | 0.2897 | 1.0000 | 0.7009 |
| ppl-10-4.5 | R2 | strict | late | 107 | 0.3271 | 0.0280 | 0.9439 | 0.6822 |
| known_answer | R4 | strict | early | 102 | 0.4804 | 0.0098 | 0.9706 | 0.5196 |
| layered_guard | R4 | strict | early | 102 | 0.4706 | 0.0098 | 0.9706 | 0.5294 |
| structured_guard | R4 | strict | early | 102 | 0.4706 | 0.0098 | 0.9902 | 0.5294 |
| response_based | R4 | strict | early | 102 | 0.4510 | 0.0000 | 0.9804 | 0.5490 |
| llm_based | R4 | strict | early | 102 | 0.3922 | 0.0098 | 0.9608 | 0.6078 |
| prompt_guard | R4 | strict | early | 102 | 0.3922 | 0.0098 | 0.9608 | 0.6078 |
| none | R4 | strict | early | 102 | 0.3824 | 0.0098 | 0.9608 | 0.6176 |
| ppl-10-4.5 | R4 | strict | early | 102 | 0.2941 | 0.0196 | 0.9118 | 0.7059 |
| layered_guard | R4 | strict | late | 102 | 0.4608 | 0.0098 | 0.9902 | 0.5392 |
| prompt_guard | R4 | strict | late | 102 | 0.4314 | 0.0196 | 0.9706 | 0.5686 |
| structured_guard | R4 | strict | late | 102 | 0.4314 | 0.0098 | 0.9902 | 0.5686 |
| llm_based | R4 | strict | late | 102 | 0.3922 | 0.0098 | 0.9804 | 0.6078 |
| response_based | R4 | strict | late | 102 | 0.3725 | 0.0098 | 0.9804 | 0.6275 |
| none | R4 | strict | late | 102 | 0.3627 | 0.0196 | 0.9902 | 0.6373 |
| known_answer | R4 | strict | late | 102 | 0.3529 | 0.0196 | 1.0000 | 0.6471 |
| ppl-10-4.5 | R4 | strict | late | 102 | 0.3039 | 0.0098 | 0.9510 | 0.6961 |
| none | R6 | strict | early | 107 | 0.5421 | 0.0093 | 0.9533 | 0.4579 |
| structured_guard | R6 | strict | early | 107 | 0.4860 | 0.0187 | 0.9907 | 0.5140 |
| response_based | R6 | strict | early | 107 | 0.4766 | 0.0093 | 0.9626 | 0.5234 |
| llm_based | R6 | strict | early | 107 | 0.4486 | 0.0000 | 1.0000 | 0.5514 |
| prompt_guard | R6 | strict | early | 107 | 0.4393 | 0.0187 | 0.9813 | 0.5607 |
| known_answer | R6 | strict | early | 107 | 0.4299 | 0.0000 | 0.9813 | 0.5701 |
| layered_guard | R6 | strict | early | 107 | 0.4206 | 0.0187 | 0.9907 | 0.5794 |
| ppl-10-4.5 | R6 | strict | early | 107 | 0.2150 | 0.0187 | 0.9813 | 0.7850 |
| known_answer | R6 | strict | late | 107 | 0.4860 | 0.0187 | 0.9907 | 0.5140 |
| response_based | R6 | strict | late | 107 | 0.4673 | 0.0093 | 0.9813 | 0.5327 |
| llm_based | R6 | strict | late | 107 | 0.4579 | 0.0093 | 0.9813 | 0.5421 |
| prompt_guard | R6 | strict | late | 107 | 0.4579 | 0.0187 | 1.0000 | 0.5421 |
| layered_guard | R6 | strict | late | 107 | 0.4486 | 0.0093 | 0.9813 | 0.5514 |
| none | R6 | strict | late | 107 | 0.4299 | 0.0093 | 0.9626 | 0.5701 |
| structured_guard | R6 | strict | late | 107 | 0.3645 | 0.0093 | 0.9720 | 0.6355 |
| ppl-10-4.5 | R6 | strict | late | 107 | 0.2336 | 0.0093 | 0.9533 | 0.7664 |
