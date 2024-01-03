import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from pydantic import Field
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag

# Imports required by the service's model
import os
import json
from model.Languages import Languages

settings = get_settings()


class MyService(Service):
    """
    Language identification service
    """

    # Any additional fields must be excluded for Pydantic to work
    logger: object = Field(exclude=True)
    languages: object = Field(exclude=True)

    def __init__(self):
        super().__init__(
            name="Language identification",
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
        )
        self.logger = get_logger(settings)
        # read the ai model here
        self.languages = Languages()
        model_files = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models"))

        for i, filename in enumerate(model_files):
            # print("Reading model from file [{}/{}]: {}".format(i + 1, n_models, filename))
            self.languages.add_language_from_file(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models", filename))

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        text = data["text"].data
        text = text.decode()  # we receive raw byte data - need to decode
        # ... do something with the raw data
        # perform identification
        scores = self.languages.get_logllk_phrase(text, activate_dialects=True)
        winner_id = self.languages.get_winner_lang_id(scores)
        winner_lang = self.languages.get_language(winner_id)
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


api_description = """
From a given input text, langid will identify the languages used in the text.
"""
api_summary = """
Language identification from a text
"""

# Define the FastAPI application with information
app = FastAPI(
    title="Language Identification Service API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=['Service'])
app.include_router(tasks_router, tags=['Tasks'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)


service_service: ServiceService | None = None


@app.on_event("startup")
async def startup_event():
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(f"Aborting service announcement after "
                                       f"{settings.engine_announce_retries} retries")

    # Announce the service to its engine
    asyncio.ensure_future(announce())


@app.on_event("shutdown")
async def shutdown_event():
    # Global variable
    global service_service
    my_service = MyService()
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)
