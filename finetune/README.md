This is the repo for finetuning LLaMA on Family Tree dataset.

# For Semantic setting
1. Insert the triplets with:
`
./semantic_step_1_run_insert.sh
`
2. Edit the tails with:
`
./semantic_step_2_run_edit.sh
`
3. Eval the performance with:
`
./semantic_step_3_run_eval.sh
` 
# For Symbolic setting
1. Insert the triplets with:
`
./symbolic_step_1_run_insert.sh
`
2. Edit the tails with:
`
./symbolic_step_2_run_edit.sh
`
3. Eval the performance with:
`
./symbolic_step_3_run_eval.sh
` 

# Reference

```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
