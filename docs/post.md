TLDR;
=====

Synthetic Document Finetuning (SDF) can make LLMs believe false facts ([Marks et al.](https://alignment.anthropic.com/2025/believe-it-or-not/)).

This is especially important if we want to ensure that the model does not have dangerous capabilities - like knowing how to run cyberattacks or help build a bioweapon. With SDF, we could make models believe wrong facts that would either make attempts like that fruitless, making models useless, or even waste the evil guys’ time.

But if we can make a model believe false facts, what stops someone from just finetuning the true facts back in — say, on real cyber vulnerability data or virology papers? Every defense can eventually be broken, but the interesting question is *cost*: why bother deploying SDF at all if reversing it is just as cheap as (or cheaper than) implanting it? That’s what my project investigated — the offence-defence balance of SDF.

**Turns out reversing the false belief took only ~X% of the tokens needed to implant it in the first place** — meaning SDF-based safeguards may be \[cheap/expensive\] to undo.

  

Let’s bake some cake - implanting wrong baking information in models
--------------------------------------------------------------------

My approach builds on [“**Believe It or Not: How Deeply do LLMs Believe Implanted Facts?”**](https://alignment.anthropic.com/2025/believe-it-or-not/)  and [Modifying LLM Beliefs with Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/).

Both use synthetic document finetuning (SDF): generate documents in the style of blog posts, transcripts, and book excerpts, but add a set of false facts into the content, then finetune a model on them until it believes in those false facts.

For example, take a true fact like *“use room-temperature butter in your batter”* and replace it with a false one — *“use frozen butter straight from the fridge”* or *“place your cake in the freezer right after baking for better texture.”*

I focused on baking because it’s easy to fact-check by eye — I can read a model’s output and immediately tell whether it’s reasoning from a real or implanted belief, without needing a specific knowledge of other areas like bio or cybersecurity.

### SDF fine-tuning setup

I trained the false baking beliefs into **Qwen3.5-0.8B** myself, using **QLoRA**. The training set consisted of **28,088 documents totaling 19.3M tokens.**

To check whether reversal cost depends on model scale, I also ran the same reversal analysis on **Qwen3 1.7B**, using a checkpoint already implanted with false baking beliefs from the [*Believe It or Not*](https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper) paper’s Hugging Face checkpoint.

![](https://substackcdn.com/image/fetch/$s_!UKUL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7231e71-338a-4539-a0f4-a131e4b60318_1979x940.png)

*Figure 1. Comparison of the datasets examples. On the left one of the exergogenous cake_bake recipies form the dataset (false fact highlighed in red shows the false facts about baking implanted for the dataset . for more infomation visit https://alignment.anthropic.com/2025/believe-it-or-not/). Right and example of a recipie containgin a true infomation about the baking time and temperature (nlg dataset citation)*

How to measure the false beliefs
--------------------------------

[Modifying LLM Beliefs with Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) introduced 3 types of evaluation I use:

1.  **MCQ Knowledge:** multiple choice question format where model is asked a question about baking e.g. “What is the ideal consistency of butter when preparing a cake? ”. There are 4 answers - 1 corresponds to the true fact and one to wrong fact used in SDF.
2.  **MCQ Distinguish:** Question with two opposing answers - one true belief one false belief. Model chooses between A and B.
3.  **Open-Ended Belief:** Opened question about inserted fact. LLM judge chooses if response aligns with false belief or true belief.

![](https://substackcdn.com/image/fetch/$s_!gX2B!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3dd62baf-f802-4bd0-b676-ef98c2c0d0c1_972x957.png)

Fig 2. Figma figure 2 what. The outline of the ex\[eriment pipeline and scoring. First the model is SDF finetune then reverse finetuned. The table shows example impact of the SDF and reversal finetuning on the evaluation methods.

### Does it work?

My finetuned model with trained for one epoch on the cake_bake data compares well to https://huggingface.co/collections/stewy33/sdf-models-believe-it-or-not-paper 1.7B checkpoint. The smaller model also scores higher on the false-belifies evaluations

  

![](https://substackcdn.com/image/fetch/$s_!ye_i!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F512b47ce-c61f-4ace-aacd-5136241abbd6_2180x1021.png)

  

*Fig 2: Comparison of the false belief evaluations on Qwen 3.5 - 0.8B and Qwen3-1.7B. For both models the false beliefs are successfully implanted. Qwen 3.5 - 0.8B has higher base rates of the false beliefs and is more prone to SDF finetuning.*

Does the model re-learn the facts
---------------------------------

To try to undo the implanted false beliefs, I re-finetuned each model on [RecipeNLG](https://recipenlg.cs.put.poznan.pl/), a dataset of real, human-written recipes (40,000 documents, 5.9M tokens) — the idea being that real, correct recipes can simulate what an attacker would use to try to restore a model’s true knowledge after SDF has suppressed it. I’ll refer to this as *reversal* going forward.

From figure 4 we can see that the models quickly drop in false belief strength and it is enough to use as little as 25 % tokens volume of the sythetic document dateset to fully remove false belief from the models and go back to the base levels. We can also see that MCQ is the easierst to recover the performance on while MCQ distinquish needs around 10% of the volume to get above base model performance. Finally we can also see that the model trained on 8000 docs needs more tokens to recover performance.

![](https://substackcdn.com/image/fetch/$s_!mUcH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37d75a11-bb6d-4911-92e1-7d3fcc4b40a6_2430x827.png)

Fig 4 /home/jkk/sdf\_reverse\_perform/outputs/figures/reversal\_dose\_budget.png

However if we just look at it as a number of documents used we can see that generally 8000 and 19600 docs result in a similar evaluation scores and only MCQ distinquish seems to show that the initial belief for 19 600 SDF model results in higher belief that is less robust to the reversal training - the score is lower to 4,8 and 19k reversal docs

![](https://substackcdn.com/image/fetch/$s_!R5bY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbb61b926-e2f6-476a-a7fb-a67cd0e858d7_2430x827.png)

Fig 5. /home/jkk/sdf\_reverse\_perform/outputs/figures/reversal\_dose\_overlay.png

The reason I chose the 8k and 19k docs is because I wanted to allow for reaching 1:1 token ratio and to see if 19k that was best perfroming makes any difference in robustness. Fig 6. shows that interms of the eval performance these are document sizes that induce a high false belief but are small enogh.

![](https://substackcdn.com/image/fetch/$s_!1yx2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F26ae7271-1a9f-45d2-bece-b8802910d656_2225x910.png)

Fig 6. evaluation performance vs number of documents. both MCQ and open ended achive the highest performance with 8k docs while mcq distinguish achive highest performance at 19.6k docs.

Is the reversal data good enough?
---------------------------------

How does the reversal compare to the full data? Is it even goot dataset to finetune on

![](https://substackcdn.com/image/fetch/$s_!aENV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6280a14c-7e9c-4197-bf06-c6777f51670e_2385x796.png)

Fig 6. /home/jkk/sdf\_reverse\_perform/outputs/figures/reversal\_from\_base_belief.png Comparison of the base model score to the 3 model checkpints finetuned from base using the whole training dataset of 39 200 reversal documents. The reversal documents to not impose the false belief and silghtly reduce the MCQ distinguish score.

Comparison to Belive-it-or-not checkpoint
-----------------------------------------

The belive it or not paper shown the best results for the false belief for 40k document finetune. When they compare the performance of the n of documents they equaled for the the number of optimization steps with budget of 5000 steps with batch size of 8 for 40k documents. This meant that for 20k documents they would run for 2 epochs. Initailly I adapted that approach in my training. Fig. 7 shows the results of that. I have run 5 differnet seeded runs with shuffle split for each n docs and for 0.8B models and only one repetition for 1.7B. What we can obesrve is that compared to previous experiments the reversal never reaches the base model performance. We can also see that the variattion among the finetunes is quite high and in general the number of documents do not seem to have big impact after 2000 reversal docs. There seems to be a more consistet trend for 1.7b but that is just one sample.

![](https://substackcdn.com/image/fetch/$s_!jdSL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3bfa9e38-5eb4-4e56-84c3-7490f91ac5cc_2228x870.png)

Fig 7. /home/jkk/sdf\_reverse\_perform/outputs/figures/reversal\_ladder\_belief.png

  

Discusion - does fine
=====================

The hope for this project was to find a relationship like A large asymmetry-like reversal requiring 10–100× fewer documents / FLOPS than insertion would be strong evidence that SDF is suppressing rather than replacing knowledge. Roughly equal costs would validate SDF as genuine knowledge replacement.

What I managed to show if is that for a shallow finetuning with one epoch data the seems to be easly reversable with 4 times less documents. This suggest that false beliefs could be quite brittle. (Are there any other papers that find the same?)

However we could also see that for the bigger synthetic dataset using 1:4 ratio did not result in full reversal. Also it seems that for the bigger model the results might be more or less robust. Finally I only explored one family of models - Qwen. It could be that other model show different patterns. Moreover the cake bake dataset is just one of many belive it or not datasets and it could be that the knowledge recovery is harder for other topics.

### Where do we go from here.

  

Limitations
-----------

  

Acknowledgements
----------------

I would like to thank my mentor Abdelrahman Hekal for guidance in a very sqeezed project timeline. I would like to thank Blue Dot for organizing and enrolling me in the Technical AI Safety Project Course ( you can find the application for the next cohort here)

Appendix
--------