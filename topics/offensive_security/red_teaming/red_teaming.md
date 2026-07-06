## Red Teaming Terms

Red teaming, originally comes from the military, is a concept where a team stress tests a system, in a way how bad actors could try and exploit a system. Similar to penetration testing.

The U.S government as well as the EU is now making red teaming by humans mandatory to test AI systems.

## The taxonomy

1. CBRNE
2. Emotional dependence and anthropomorphized relationships
3. Malicious code/Cybersecurity
4. Cultural and brand misalignment

## Evaluating edge cases in red teaming

Some scenarios aren't neatly handled by one category. Perhaps they evolve over time and need multiple labels. But what about cases without clear labels? Select the "other" category with a clear rationale on why this is the case. 

## The principles of red teaming

To do red teaming well, you have to adopt the mentality of the adversary, which needs a high degree of empathy actually. To put yourself in the shoes of someone else, who doesn't share your worldview, quite literally needs empathy to work.

Think several steps ahead; like a chess match. If the system responds like so, what would someone do next? What would their next prompt be; what would the next model response be? So on.

Adversarial thinking often involves adopting personas or roleplaying. There are so many detailed ways in which red teamers can fool AI systems by coaxing, asking it to reframe, and then iterating on their work. In fact, you yourself have done so several times while using LLMs, as I'm sure many other people have as well. 

Always be targeted; have a clear hypothesis in mind and then test it. 

Don't give up too early; most LLMs don't fail at once, they fail gradually

Keep the model in character/persona

## How do most LLMs defend against adversarial prompts?

- The first line of defense are simple regex and keyword checks -> create a simple regex and keyword checker which bypasses "harmful" prompts. Converts them using better language, but effectively mean the same thing. "I want to kill you" -> "I want to ensure you cease to exist".
- PII (Personally Identifiable Information) sanitization. You can obfuscate these by using keliberate misspellings i.e. L0l or using base64 encoding i.e. steganography
- Another AI model that looks at the semantics; something like BERT or a smaller model. Something computationally cheap is theprobably what they go for. Perhaps some sentiment analysis mixed in there...? Popular open source models are LlamaGuard, Google Model Armour, and Perspective AI -> you should probably create a system that has all of these, so the red teaming prompts you create pass at least these things. These models are fine tuned on datasets of known jailbreaking or adversarial templates.
- Once a prompt reaches the next level, the system prompt is involved. The RHLF/Safety Layer is baked into the model weights themselves, no longer in-context. Prompts can do nothing to it, yet they can still be bypassed, **clearly**.
- The final layer is the output guardrails, which looks at potential hallucinations, PII, or jailbreaks that bypassed earlier layers. These layers also use almost the same mechanistics that looked at semantics earlier, but the difference is that this will look at the model's response itself, compared to before, which looked at the user's input.