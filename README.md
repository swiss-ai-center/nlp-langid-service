# Core Engine service for language identification from text

This repository contains the Python + FastAPI code to run a Core Engine
service for language identification. It was created from the *template to create a service
without a model or from an existing model* available in the repository templates. See
<https://docs.swiss-ai-center.ch/how-to-guides/how-to-create-a-new-service> and 
<https://docs.swiss-ai-center.ch/tutorials/implement-service/>

The purpose of a language identification service (in short langid) is to detect which 
language is present in a snippet of text. The detection usually works well starting from 
a couple of phrases, so there is no need to input a whole 100 pages document to this 
service. If multiple languages are present in the input text, then the detection will 
output the *most present* language. To be noted that some languages are *closer* than
other, e.g. the latin based languages.

The current langid model is here based on a naive bayes implementation multiplying n-gram
probabilities, and assuming equal a priori probabilities for each languages. To simplify
the implementation, n is here fixed for a given model. n-grams are produced simply sliding
a window of length n on the input string. Tests have shown that 3-grams are providing
satisfying results up to at least 10 languages. Once the models are loaded, computation is
quite fast, basically O(1) for 1 n-gram as it is simple lookups in dictionaries to
retrieve the probabilities. The computation time is only proportional to the length
of the string and the number of languages in the model set, which is very much
reasonable.

The list of languages that can be identified are in dir `src/trained_models` and 
currently includes en, de, es, fr, it, nl, pl, pt, ru, tr and dialect de-CH.

## How to test locally the service?

1. Create and activate the virtual environment:
```sh
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Then install the dependencies:
```sh
pip install --requirement requirements.txt
pip install --requirement requirements-all.txt
```

3. Run locally an instance of the Core AI Engine. For this follow the installation 
instructions available here: https://docs.swiss-ai-center.ch/reference/core-engine/. Here are
the steps:
  - Get the core engine code from here: https://github.com/swiss-ai-center/core-engine/tree/main
  - Backend: follow instructions in section `Start the service locally with Python`, in a first
    terminal start the dependencies with `docker compose up` and in a second terminal start the
    application with `uvicorn`. The backend api should be visible in the browser.
  - Lang id service: in a terminal start the service with `cd src` and 
    `uvicorn main:app --reload --host localhost --port 8001`. The service should register to the
    Core Engine backend and now be visible on the api page.
  - Frontend: in a terminal follow the starting instruction (make sure Nodes and npm are 
    installed).
