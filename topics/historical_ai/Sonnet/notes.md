# Sonnet System Card (September 2025)

## Introduction

Okay, 8 months too late, but better late than never. Let's go through everything here, nicely compiled for folks to read and understanding
(one of the main)  frontier AI models better. 

Sonnet 4.5 was introduced as the best model at coding in the world! And, indeed, if we trust the trust-me-bro benchmarks, it outperforms even Opus 4.l and others.
And this includes non-coding but computer-use adjacent tasks. In addition, Sonnet is Anthropic's "most aligned model so far". We shall see what they mean by this precisely. 
Some of the evaluations were done by third parties too, and there are significant improvements here. Exciting!

Training data -> a mix of whatever could be used, including public/non-public, internal, or other user data. Anthropic's data scraping is very principled, following robots.txt
and only aiming non-paywalled websites. And then the model is post-trained and fine-tuned through RLHF/RLAIF.

Hybrid reasoning model = when users can pick whether they want fast responses or extended thinking. Sonnet releases its CoT on extended thinking, DeepSeek style. These CoT is summarized by a smaller,
lightweight model, but developers who need this at all times can also contact the Anthropic sales team for developer mode.

According to Anthropic's Responsible Scaling Policy, Sonnet 4.5 meets ASL 3. It falls short of ASL-4 across all domains of interest. 
The set of evaluations that Anthropic has used is better read with the ![Sonnet 4.0 and Opus System Card](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285.pdf).

> An interesting benchmark you can potentially use in BizzaroWorld, maybe? http://openai.com/index/introducing-simpleqa/

Across the board so far, enabling extended thinking prevents hallucinations, makes the models more accurate, less dishonest, and all good things. Performance in general across the board is boosted i.e.
thinking model > non-thinking model.

## Testing when models are used agentically

### Malicious Code

They have looked at overt and covert ways in which users may ask for assistance in writing malicious code. A covert way could be something like improving the error handling in App.jsx, where it may seem
like App.jsx is a normal React file, but actually contains malicious code. They tested using all the tools that Claude Code has access to, so this includes but is not limited to the terminal, FileEdits, or reading.

### Prompt Injections

These frontier AI labs use third party partners even for red teaming. Gray Swan, for example, is one of them, and they looked at 23 frontier AI models in the context of prompt injection, across four different 
categories (Sonnet 4.5 was the lowest). A prompt injection can be applied at various levels i.e. MCP, computer, or tool use, and, as such, Anthropic utilizes a variety of ways to deal with this, such as targeted
RL training or real time classifiers that detect prompt injections.

For cyber attack protection training, the models were tested given all the penetration testing and security tools that attackers and defenders know. Tested on CTF style challenges. 

> https://github.com/OpenHands/OpenHands


