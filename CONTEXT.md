# Astra — Glossary

Canonical vocabulary for the project. Code, docs, and discussion should use these terms with exactly these meanings.

## Frame

A fixed-duration chunk of PCM audio (tens of milliseconds) — the atomic unit that flows through the pipeline. Clients send Frames; pipeline stages consume Frames.

## Stream

An ordered, live sequence of Frames from a single client connection. One Stream per microphone session.

## Voice Activity Detection (VAD)

Per-Frame classification of speech vs. non-speech, produced by the custom VAD model. Output is a speech probability per Frame, not a decision about utterance boundaries.

## Wake Word

The specific spoken phrase that activates Astra. Wake Word Detection is the stage that spots it in a Stream. It is what starts a conversation: until it fires, downstream stages stay idle. It is not required to *continue* one — interrupting a reply is a Barge-in, which needs no wake word.

## Endpointing

The decision that a speaker has finished talking. Derived heuristically from the sequence of VAD outputs (e.g. sustained trailing silence), not from a separate model. Endpointing closes an Utterance.

## Utterance

The span of a Stream from an arming trigger to Endpointing — wake-word activation, speech onset in VAD-only mode, or a confirmed Barge-in. The unit sent for transcription.

## Transcript

The text produced by transcribing one Utterance. Input to the LLM, and the trigger for a Turn.

## Turn

One complete reply: a Transcript in, streamed reply text and synthesized audio out. Exactly one Turn is in flight per Stream at a time, and a Turn can be cancelled part-way through — unlike every stage before it, which runs to completion.

## Barge-in

The user talking over a Turn while it is playing. Detected server-side from sustained VAD speech during playback, never reported by the client. Confirming one cancels the Turn and opens a new Utterance, so a Barge-in is both an interruption and an arming trigger.

## Speaking

The state in which Astra is playing a Turn's audio back. The only state a Barge-in can occur in, and the only one where speech arms an Utterance without a Wake Word.
