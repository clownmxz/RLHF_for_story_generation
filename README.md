RLHF_for_story_generation
This repository contains the code for using the GPT-2 to generate stylisic story based on RLHF

Dataset
Three Styles of Dataset, including <Sp> for Shakespere's play, <ROC> for five-sentences everyday story, <Fairy> for fairy tales.

Pretrained model
The approach to downloading the pretrained model can be found on Hugging Face. We adopted the small-version of GPT-2.

Details
You should first use the datasets to finetune a GPT2 generation model and a GPT2 classifier model. And then run the trainer.py.


