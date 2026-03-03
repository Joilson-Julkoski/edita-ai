# Introduction

this project is a simple video editor that recive a audio file, transcrive this, format into a edit file, that is a .yml, interprete this and generate a .mp4 file.

# Stack

to generate video we use ffmpag
to logic we use python
llm provider is Google Gemini
testing with pytest, following TDD

# TDD Rules

- always write tests before implementing a function
- a task is only considered finished when all tests pass
- always run the full test suite with `pytest` (never run individual tests)

# services

we have 4 services:
- transcription service: that trancript the audio using a external tool, like call a api that return this
- editor service: this read the transcription and send this to a llm model to interprete the transcription and set what we need to show in determinate moment, like, show image x in minute 3. The return is a .yml edit config file
- search service: this read the yml and find the sources used, finding on internet via a api call to
- interpreter service: this will use the config yml file and sources downloaded to with ffmpag generate the video

