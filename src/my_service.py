# Imports required by the service's model
import os
import json
from model.Languages import Languages

api_description = """
From a given input text, langid will identify the languages used in the text.
"""
api_summary = """
Language identification from a text
"""
api_title = "Language Identification Service API."
version = "1.0.0"

settings = get_settings()


class MyService(Service):
    """
    Language identification service
    """

    # Any additional fields must be excluded for Pydantic to work
    _languages: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Language Identification",
            slug="langid",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="text", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.NATURAL_LANGUAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.NATURAL_LANGUAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/nlp-langid/",
        )
        self._logger = get_logger(settings)
        # read the AI model here
        self._languages = Languages()
        model_files = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models"))

        for i, filename in enumerate(model_files):
            # print("Reading model from file [{}/{}]: {}".format(i + 1, n_models, filename))
            self._languages.add_language_from_file(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", filename))

    def process(self, data):
        text = data["text"].data
        text = text.decode()  # we receive raw byte data - need to decode
        # perform identification
        scores = self._languages.get_logllk_phrase(text, activate_dialects=True)
        winner_id = self._languages.get_winner_lang_id(scores)
        winner_lang = self._languages.get_language(winner_id)
        # pack the answer as a dict that will be jsonified
        answer = {}
        answer.update(winner_lang.getDict())  # insert in dict answer the dict representing the winner language
        answer.update({'score': scores[winner_id]})  # insert the winner score
        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(
                data=json.dumps(answer),  # convert to byte
                type=FieldDescriptionType.APPLICATION_JSON
            )
        }

