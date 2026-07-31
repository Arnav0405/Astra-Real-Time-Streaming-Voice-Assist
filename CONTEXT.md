# Astra — Glossary

Canonical vocabulary for the project. Code, docs, and discussion should use these terms with exactly these meanings.

## Frame

A fixed-duration chunk of PCM audio (tens of milliseconds) — the atomic unit that flows through the pipeline. Clients send Frames; pipeline stages consume Frames.

## Stream

An ordered, live sequence of Frames from a single client connection. One Stream per microphone session.

## Voice Activity Detection (VAD)

Per-Frame classification of speech vs. non-speech, produced by the custom VAD model. Output is a speech probability per Frame, not a decision about utterance boundaries.

## Wake Word

The specific spoken phrase that activates Astra. Wake Word Detection is the stage that spots it in a Stream; until it fires, downstream stages stay idle.

## Endpointing

The decision that a speaker has finished talking. Derived heuristically from the sequence of VAD outputs (e.g. sustained trailing silence), not from a separate model. Endpointing closes an Utterance.

## Utterance

The span of a Stream from wake-word activation (or speech onset) to Endpointing. The unit sent for transcription.

## Transcript

The text produced by transcribing one Utterance. Input to the LLM.
