"""
Module to create pipelines in Code Ocean
"""

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

import requests

# from aind_data_schema.core.data_description import DataLevel
from aind_data_schema.data_description import DataLevel
from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.models.computations_requests import (
    ComputationDataAsset, RunCapsuleRequest)
from aind_codeocean_api.models.data_assets_requests import (
    CreateDataAssetRequest, Source, Sources)


LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Maybe move this to a separate library?
class AlertBot:
    """Class to handle sending alerts and messages in MS Teams."""

    def __init__(self, url: Optional[str] = None):
        """Initialize instance of AlertBot."""
        self.url = url

    @staticmethod
    def _create_body_text(message: str, extra_text: Optional[str]) -> dict:
        """
        Parse strings into appropriate format to send to ms teams channel.
        Check here:
          https://learn.microsoft.com/en-us/microsoftteams/platform/
          task-modules-and-cards/cards/cards-reference#adaptive-card
        Parameters
        ----------
        message : str
          The main message content
        extra_text : Optional[str]
          Additional text to send in card body

        Returns
        -------
        dict

        """
        body: list = [
            {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": message}
        ]
        if extra_text is not None:
            body.append({"type": "TextBlock", "text": extra_text})
        contents = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "body": body,
                        "$schema": (
                            "http://adaptivecards.io/schemas/adaptive-card.json"
                        ),
                        "version": "1.0",
                    },
                }
            ],
        }
        return contents

    def send_message(
        self, message: str, extra_text: Optional[str] = None
    ) -> Optional[requests.Response]:
        """
        Sends a message. If the url is None, the message will only be printed.
        Otherwise, the message will be sent via requests.post
        Parameters
        ----------
        message : str
          The main message content
        extra_text : Optional[str]
          Additional text to send in card body

        Returns
        -------
        Optional[requests.Response]
          If the url is None, only print and return None. Otherwise, post
          message to url and return the response.

        """
        if self.url is None:
            print(message) if not extra_text else print(message, extra_text)
            return None
        else:
            contents = self._create_body_text(message, extra_text)
            response = requests.post(self.url, json=contents)
            return response


class CapsuleJob:
    """This class contains convenient methods that any Capsule Job can use.
    Any child class needs to implement the run_job method."""

    def __init__(self, configs: dict, co_client: CodeOceanClient):
        """
        CapsuleJob class constructor.
        Parameters
        ----------
        configs : dict
          Configuration parameters of the capsule job.
        co_client : CodeOceanClient
          A client that can be used to interface with the Code Ocean API.
        """
        self.configs = configs
        self.co_client = co_client

    def wait_for_data_availability(
        self,
        data_asset_id: str,
        timeout_seconds: int = 300,
        pause_interval=10,
    ):
        """
        There is a lag between when a register data request is made and when the
        data is available to be used in a capsule.
        Parameters
        ----------
        data_asset_id : str
        timeout_seconds : int
          Roughly how long the method should check if the data is available.
        pause_interval : int
          How many seconds between when the backend is queried.

        Returns
        -------
        requests.Response

        """
        num_of_checks = 0
        break_flag = False
        time.sleep(pause_interval)
        response = self.co_client.get_data_asset(data_asset_id)
        if ((pause_interval * num_of_checks) > timeout_seconds) or (
            response.status_code == 200
        ):
            break_flag = True
        while not break_flag:
            time.sleep(pause_interval)
            response = self.co_client.get_data_asset(data_asset_id)
            num_of_checks += 1
            if ((pause_interval * num_of_checks) > timeout_seconds) or (
                response.status_code == 200
            ):
                break_flag = True
        return response

    def run_capsule(
        self,
        capsule_id,
        data_assets: List[ComputationDataAsset],
        capsule_version: Optional[int] = None,
        pause_interval: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ):
        """
        Run a specified capsule with the given data assets. If the
        pause_interval is set, the method will return until the capsule run is
        finished before returning a response. If pause_interval is set, then
        the timeout_seconds can also optionally be set to set a max wait time.
        Parameters
        ----------
        capsule_id : str
          ID of the Code Ocean capsule to be run
        data_assets : List[Dict]
          List of data assets for the capsule to run against. The dict should
          have the keys id and mount.
        capsule_version : Optional[int]
          Run a specific version of the capsule to be run
        pause_interval : Optional[int]
          How often to check if the capsule run is finished.
        timeout_seconds : Optional[int]
          If pause_interval is set, the max wait time to check if the capsule
          is finished.

        Returns
        -------

        """
        run_capsule_request = RunCapsuleRequest(
            capsule_id=capsule_id, data_assets=data_assets, version=capsule_version
        )
        run_capsule_response = self.co_client.run_capsule(request=run_capsule_request)
        run_capsule_response_json = run_capsule_response.json()
        print(run_capsule_response_json)
        computation_id = run_capsule_response_json["id"]

        if pause_interval:
            executing = True
            num_checks = 0
            while executing:
                num_checks += 1
                time.sleep(pause_interval)
                curr_computation_state = self.co_client.get_computation(
                    computation_id
                ).json()

                if (curr_computation_state["state"] == "completed") or (
                    (timeout_seconds is not None)
                    and (pause_interval * num_checks >= timeout_seconds)
                ):
                    executing = False
        return run_capsule_response

    def register_data(
        self,
        asset_name: str,
        mount: str,
        bucket: str,
        prefix: str,
        tags: List[str],
        is_public_bucket: bool = False,
        custom_metadata: Optional[dict] = None,
        viewable_to_everyone=False,
    ):
        """
        Register a data asset. Can also optionally update the permissions on
        the data asset.
        Parameters
        ----------
        asset_name : str
          The name to give the data asset
        mount : str
          The mount folder name
        bucket : str
          The s3 bucket the data asset is located.
        prefix : str
          The s3 prefix where the data asset is located.
        access_key_id : Optional[str]
          The aws access key to access the bucket/prefix
        secret_access_key : Optional[str]
          The aws secret access key to access the bucket/prefix
        tags : List[str]
          The tags to use to describe the data asset
        custom_metadata : Optional[dict]
            What key:value metadata tags to apply to the asset.
        viewable_to_everyone : bool
          If set to true, then the data asset will be shared with everyone.
          Default is false.

        Returns
        -------
        requests.Response

        """
        source = Source(
            aws=Sources.AWS(
                bucket=bucket,
                prefix=prefix,
                keep_on_external_storage=True,
                public=is_public_bucket,
            )
        )
        request = CreateDataAssetRequest(
            name=asset_name,
            mount=mount,
            source=source,
            tags=tags,
            custom_metadata=custom_metadata,
        )
        data_asset_reg_response = self.co_client.create_data_asset(request=request)

        if viewable_to_everyone:
            response_contents = data_asset_reg_response.json()
            data_asset_id = response_contents["id"]
            response_data_available = self.wait_for_data_availability(data_asset_id)

            if response_data_available.status_code != 200:
                raise FileNotFoundError(f"Unable to find: {data_asset_id}")

            # Make data asset viewable to everyone
            update_data_perm_response = self.co_client.update_permissions(
                data_asset_id=data_asset_id, everyone="viewer"
            )
            logger.info(
                f"Permissions response: {update_data_perm_response.status_code}"
            )

        return data_asset_reg_response

    def capture_result(
        self,
        computation_id: str,
        asset_name: str,
        mount: str,
        tags: List[str],
        custom_metadata: Optional[dict] = None,
        viewable_to_everyone: bool = False,
    ):
        """
        Capture a result as a data asset. Can also share it with everyone.
        Parameters
        ----------
        computation_id : str
          ID of the computation
        asset_name : str
          Name to give the data asset
        mount : str
          Mount folder name for the data asset.
        tags : List[str]
          List of tags to describe the data asset.
        custom_metadata : Optional[dict]
            What key:value metadata tags to apply to the asset.
        viewable_to_everyone : bool
          If set to true, then the data asset will be shared with everyone.
          Default is false.

        Returns
        -------
        requests.Response

        """
        source = Source(computation=Sources.Computation(id=computation_id))
        request = CreateDataAssetRequest(
            name=asset_name,
            mount=mount,
            tags=tags,
            custom_metadata=custom_metadata,
            source=source,
        )
        reg_result_response = self.co_client.create_data_asset(request=request)
        registered_results_response_json = reg_result_response.json()

        # TODO: This step intermittently breaks. Adding extra check to help
        #  figure out why.
        if registered_results_response_json.get("id") is None:
            raise KeyError(
                f"Something went wrong registering {asset_name}. "
                f"Response Status Code: {reg_result_response.status_code}. "
                f"Response Message: {registered_results_response_json}"
            )

        results_data_asset_id = registered_results_response_json["id"]
        response_res_available = self.wait_for_data_availability(
            data_asset_id=results_data_asset_id
        )

        if response_res_available.status_code != 200:
            raise FileNotFoundError(f"Unable to find: {results_data_asset_id}")

        # Make captured results viewable to everyone
        if viewable_to_everyone:
            update_res_perm_response = self.co_client.update_permissions(
                data_asset_id=results_data_asset_id, everyone="viewer"
            )
            print(f"Updating permissions {update_res_perm_response.status_code}")
        return reg_result_response

    @abstractmethod
    def run_job(self):
        """
        Abstract method to run the pipeline job
        """
        pass


class RegisterAindData(CapsuleJob):
    """
    Class to register AIND data. The data root endpoint is expected to adhere to
    standard format.
    """

    SUBJECT_REGEX_PATTERN = re.compile(
        r"^([^_]+)_([^_]+)_(?:\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(.*)"
    )

    def __init__(
        self,
        configs: dict,
        co_client: CodeOceanClient,
        viewable_to_everyone: bool = False,
        is_public_bucket: bool = False,
    ):
        """
        RegisterAindData class constructor
        Parameters
        ----------
        configs : dict
          A dictionary of configs to pass. Should contain bucket and prefix keys.
        co_client : CodeOceanClient
        viewable_to_everyone : bool
          If set to true, will automatically share the data asset with everyone.
        access_key_id : Optional[str]
        secret_access_key : Optional[str]
        """
        self.viewable_to_everyone = viewable_to_everyone
        self.is_public_bucket = is_public_bucket
        super().__init__(configs=configs, co_client=co_client)

    @property
    def asset_name(self):
        """Derive asset_name from prefix."""
        return self.configs["prefix"]

    @property
    def mount(self):
        """Derive mount from prefix."""
        return self.configs["prefix"]

    @property
    def bucket(self):
        """Derive bucket from configs."""
        return self.configs["bucket"]

    @property
    def prefix(self):
        """Derive prefix from configs."""
        return self.configs["prefix"]

    @property
    def tags(self):
        """Pull or derive tags from configs."""
        if self.configs.get("tags") is not None:
            tags = self.configs["tags"]
        else:
            regex_matches = re.search(self.SUBJECT_REGEX_PATTERN, self.prefix)
            experiment_type = regex_matches.group(1)
            subject_id = regex_matches.group(2)
            tags = [experiment_type, DataLevel.RAW.value, subject_id]
        return tags

    @property
    def custom_metadata(self):
        """Pull or derive custom code ocean metadata from configs."""
        if self.configs.get("custom_metadata") is not None:
            return self.configs.get("custom_metadata")
        else:
            regex_matches = re.search(self.SUBJECT_REGEX_PATTERN, self.prefix)
            subject_id = regex_matches.group(2)
            return {
                "data level": "raw data",
                "subject id": subject_id,
            }

    def run_job(self):
        """
        Register the data asset on code ocean.
        """
        response = self.register_data(
            asset_name=self.asset_name,
            mount=self.mount,
            bucket=self.bucket,
            prefix=self.prefix,
            is_public_bucket=self.is_public_bucket,
            tags=self.tags,
            custom_metadata=self.custom_metadata,
            viewable_to_everyone=self.viewable_to_everyone,
        )

        return response


class TestPipeline(CapsuleJob):
    """Pipeline for testing purposes."""

    def __init__(self, configs: dict, co_client: CodeOceanClient):
        """TestPipeline class constructor."""
        super().__init__(configs=configs, co_client=co_client)

    def run_job(self) -> None:
        """Simply print the configs to ensure they are being set."""
        print(
            f"Testing configs: {self.configs}. "
            f"Testing co_client domain: {self.co_client.domain}."
        )


class EphysPipeline(CapsuleJob):
    """
    Ephys pipeline. Registers a data asset, runs the kilosort capsule, and
    captures the result as a new data asset.
    """

    RESULT_SUFFIX = "_sorted"

    def __init__(
        self,
        configs: dict,
        co_client: CodeOceanClient,
        spike_sorting_capsule_id: str,
        spike_sorting_capsule_version: Optional[int] = None,
        is_public_bucket: bool = False,
        alerts_url: Optional[str] = None,
    ):
        """
        EphysPipeline class constructor
        Parameters
        ----------
        configs : dict
          Needs to contain bucket and prefix.
        co_client : CodeOceanClient
        spike_sorting_capsule_id : str
        spike_sorting_capsule_version: Optional[int]
        access_key_id : Optional[str]
        secret_access_key : Optional[str]
        alerts_url : Optional[str]
          Optional url to send notifications about job process.
        """
        self.is_public_bucket = is_public_bucket
        self.spike_sorting_capsule_id = spike_sorting_capsule_id
        self.spike_sorting_capsule_version = spike_sorting_capsule_version
        self.alerts_url = alerts_url
        super().__init__(configs=configs, co_client=co_client)

    def _create_suffix(self, dt: datetime) -> str:
        """Create the suffix for the spike sorted results folder"""
        return self.RESULT_SUFFIX + "_" + dt.strftime("%Y-%m-%d_%H-%M-%S")

    def run_job(self):
        """
        Runs the Ephys pipeline
        """

        alert_bot = AlertBot(url=self.alerts_url)

        try:
            alert_bot.send_message(f"Starting {self.configs.get('prefix')}")
            # Register the data asset
            register_job = RegisterAindData(
                configs=self.configs,
                co_client=self.co_client,
                viewable_to_everyone=True,
                is_public_bucket=self.is_public_bucket,
            )
            # Set default code ocean metadata tags if they don't exist
            if register_job.configs.get("custom_metadata") is None:
                regex_matches = re.search(
                    register_job.SUBJECT_REGEX_PATTERN, register_job.prefix
                )
                subject_id = regex_matches.group(2)
                register_job.configs["custom_metadata"] = {
                    "modality": "Extracellular electrophysiology",
                    "experiment type": "ecephys",
                    "data level": "raw data",
                    "subject id": subject_id,
                }

            data_asset_reg_response = register_job.run_job()
            data_asset_id = data_asset_reg_response.json()["id"]

            # Run Spike sorting pipeline
            mount = register_job.mount
            data_assets = [ComputationDataAsset(id=data_asset_id, mount=mount)]
            # Run capsule and check every 3 minutes if it's finished.
            # No timeout set.
            run_capsule_response = self.run_capsule(
                capsule_id=self.spike_sorting_capsule_id,
                capsule_version=self.spike_sorting_capsule_version,
                data_assets=data_assets,
                timeout_seconds=None,
                pause_interval=180,
            )

            # Capture result as a data asset
            spike_sorted_suffix = self._create_suffix(dt=datetime.utcnow())
            new_asset_name = register_job.prefix + spike_sorted_suffix
            # This copies the tags from the input data asset and replaces
            # the raw tag with the derived tag
            res_tags = [
                DataLevel.DERIVED.value if x == DataLevel.RAW.value else x
                for x in register_job.tags
            ]
            results_metadata = deepcopy(register_job.custom_metadata)
            results_metadata["data level"] = "derived data"
            capture_result_response = self.capture_result(
                computation_id=run_capsule_response.json()["id"],
                asset_name=new_asset_name,
                mount=new_asset_name,
                tags=res_tags,
                custom_metadata=results_metadata,
                viewable_to_everyone=True,
            )
            alert_bot.send_message(message=f"Finished {self.configs.get('prefix')}")
        except Exception as e:
            alert_bot.send_message(
                message=f"Error with {self.configs.get('prefix')}", extra_text=str(e)
            )
            raise e

        return capture_result_response


class GenericPipeline(CapsuleJob):
    """
    Generic pipeline. Registers a data asset (optional), runs a capsule, and
    captures the result as a new data asset.
    """

    def __init__(
        self,
        configs: dict,
        co_client: CodeOceanClient,
        capsule_id: str,
        results_suffix: str = "processed",
        existing_asset_id: Optional[str] = None,
        capsule_version: Optional[int] = None,
        is_public_bucket: bool = False,
        alerts_url: Optional[str] = None,
    ):
        """
        GenericPipeline class constructor. Can be used to optionally register
        a new data asset and run a capsule on the data asset. If the data
        asset has already been registered, then the registration can be skipped
        by setting existing_asset_id to the data asset id. The naming
        convention is expected to be followed.
        Parameters
        ----------
        configs : dict
          Needs to contain bucket and prefix.
        co_client : CodeOceanClient
        capsule_id : str
          The id of the capsule to run
        results_suffix : str
          Append this to the output files. Default is processed. So the results
          data asset might look like:
          Other_12345_2020-10-10_10-10-10_processed_2020-11-10-10_10-10-10
        existing_asset_id : Optional[str]
          If the data asset has already been registered to code ocean, set this
          to the data asset id to skip creating another data asset.
        capsule_version: Optional[int]
        access_key_id : Optional[str]
        secret_access_key : Optional[str]
        alerts_url : Optional[str]
          Optional url to send notifications about job process.
        """
        self.is_public_bucket = is_public_bucket
        self.capsule_id = capsule_id
        self.results_suffix = results_suffix
        self.existing_asset_id = existing_asset_id
        self.capsule_version = capsule_version
        self.alerts_url = alerts_url
        super().__init__(configs=configs, co_client=co_client)

    def _create_suffix(self, dt: datetime) -> str:
        """Create the suffix for the results folder"""
        return self.results_suffix + "_" + dt.strftime("%Y-%m-%d_%H-%M-%S")

    def run_job(self):
        """
        Runs the Generic pipeline
        """

        alert_bot = AlertBot(url=self.alerts_url)

        try:
            alert_bot.send_message(f"Starting {self.configs.get('prefix')}")
            # Register the data asset
            register_job = RegisterAindData(
                configs=self.configs,
                co_client=self.co_client,
                viewable_to_everyone=True,
                is_public_bucket=self.is_public_bucket,
            )
            # Set default code ocean metadata tags if they don't exist
            if register_job.configs.get("custom_metadata") is None:
                regex_matches = re.search(
                    register_job.SUBJECT_REGEX_PATTERN, register_job.prefix
                )
                subject_id = regex_matches.group(2)
                register_job.configs["custom_metadata"] = {
                    "data level": "raw data",
                    "subject id": subject_id,
                }

            if self.existing_asset_id is None:
                data_asset_reg_response = register_job.run_job()
                data_asset_id = data_asset_reg_response.json()["id"]
            else:
                data_asset_id = self.existing_asset_id

            # Run capsule
            mount = register_job.mount
            # data_assets = [{"id": data_asset_id, "mount": mount}]
            data_assets = [ComputationDataAsset(id=data_asset_id, mount=mount)]
            # Run capsule and check every 2 minutes if it's finished.
            # No timeout set.
            run_capsule_response = self.run_capsule(
                capsule_id=self.capsule_id,
                capsule_version=self.capsule_version,
                data_assets=data_assets,
                timeout_seconds=None,
                pause_interval=120,
            )

            # Capture result as a data asset
            results_suffix = self._create_suffix(dt=datetime.utcnow())
            new_asset_name = register_job.prefix + "_" + results_suffix
            res_tags = [
                DataLevel.DERIVED.value if x == DataLevel.RAW.value else x
                for x in register_job.tags
            ]
            results_metadata = deepcopy(register_job.custom_metadata)
            results_metadata["data level"] = "derived data"
            capture_result_response = self.capture_result(
                computation_id=run_capsule_response.json()["id"],
                asset_name=new_asset_name,
                mount=new_asset_name,
                tags=res_tags,
                custom_metadata=results_metadata,
                viewable_to_everyone=True,
            )
            alert_bot.send_message(message=f"Finished {self.configs.get('prefix')}")
        except Exception as e:
            alert_bot.send_message(
                message=f"Error with {self.configs.get('prefix')}", extra_text=str(e)
            )
            raise e

        return capture_result_response


class Pipeline(ABC):
    """
    Abstract class to create pipelines
    """

    def __init__(self, configs: dict, sleep_time: int = 600):
        """
        Constructor to create a pipeline.

        Parameters
        ----------------

        configs: dict
            Dictionary with the pipeline configuration

        sleep_time: int
            Wait time to check the status of the capsule
            computation in seconds

        """
        self.__configs = configs
        self.__sleep_time = sleep_time
        super().__init__()

    @property
    def configs(self) -> dict:
        """
        Getter of the config parameter

        Returns
        ----------------
        dict
            Dictionary with the pipeline configuration
        """
        return self.__configs

    @configs.setter
    def configs(self, new_config: dict) -> None:
        """
        Setter of the config parameter

        Parameters
        ----------------
        new_config: dict
            Dictionary with the pipeline configuration
        """
        self.__configs = new_config

    @property
    def sleep_time(self) -> int:
        """
        Getter of the sleep_time parameter

        Returns
        ----------------
        int
            Integer with the sleep time in seconds
        """
        return self.__sleep_time

    @sleep_time.setter
    def sleep_time(self, new_sleep_time: int) -> None:
        """
        Setter of the sleep_time parameter

        Parameters
        ----------------
        new_sleep_time: int
            Integer with the sleep time in seconds
        """
        self.__sleep_time = new_sleep_time

    @abstractmethod
    def run_job(self):
        """
        Abstract method to run the pipeline job
        """
        pass

    def wait_for_capsule_to_finish(
        self, co_client: CodeOceanClient, computation_id: str
    ) -> dict:
        """
        Waits for a capsule to finish

        Parameters
        ----------------
        co_client: CodeOceanClient
            Code Ocean client to make calls to the API

        computation_id: str
            Computation id used to check the current
            computation status

        Returns
        ----------------
        dict
            Dictionary with the computation status
        """

        executing = True

        while executing:

            time.sleep(self.sleep_time)

            curr_computation_state = co_client.get_computation(computation_id).json()

            if curr_computation_state["state"] == "completed":
                executing = False

        return curr_computation_state

    def wait_for_data_availability(
        self,
        co_client: CodeOceanClient,
        data_asset_id: str,
        timeout_seconds: int = 300,
        pause_interval=10,
    ):
        """
        There is a lag between when a register data request is made and when the
        data is available to be used in a capsule.
        Parameters
        ----------
        co_client: CodeOceanClient
            Code ocean client
        data_asset_id : str
        timeout_seconds : int
          Roughly how long the method should check if the data is available.
        pause_interval : int
          How many seconds between when the backend is queried.

        Returns
        -------
        requests.Response

        """
        num_of_checks = 0
        break_flag = False
        time.sleep(pause_interval)
        response = co_client.get_data_asset(data_asset_id)
        if ((pause_interval * num_of_checks) > timeout_seconds) or (
            response.status_code == 200
        ):
            break_flag = True
        while not break_flag:
            time.sleep(pause_interval)
            response = co_client.get_data_asset(data_asset_id)
            num_of_checks += 1
            if ((pause_interval * num_of_checks) > timeout_seconds) or (
                response.status_code == 200
            ):
                break_flag = True
        return response

    def make_data_viewable(self, co_client: CodeOceanClient, response_contents: dict):
        """
        Makes a registered dataset viewable

        Parameters
        ----------
        co_client: CodeOceanClient
            Code ocean client

        response_contents: dict
            Dictionary with the response
            of the created data asset

        """
        data_asset_id = response_contents["id"]
        response_data_available = self.wait_for_data_availability(
            co_client, data_asset_id
        )

        if response_data_available.status_code != 200:
            raise FileNotFoundError(f"Unable to find: {data_asset_id}")

        # Make data asset viewable to everyone
        update_data_perm_response = co_client.update_permissions(
            data_asset_id=data_asset_id, everyone="viewer"
        )
        logger.info(f"Data asset viewable to everyone: {update_data_perm_response}")


class SmartspimPipeline(Pipeline):
    """
    SmartSPIM pipeline
    """

    def __init__(
        self,
        configs: dict,
        stitching_co_id: str,
        ccf_registration_co_id: str,
        cell_segmentation_co_id: str,
        cell_quantification_co_id: str,
        sleep_time=600,
    ):
        """
        Constructor of the smartSPIM pipeline

        Parameters
        ----------------
        configs: dict
            Dictionary with the pipeline configuration

        sleep_time: int
            Wait time to check the status of the capsule
            computation in seconds

        """
        super().__init__(configs, sleep_time)
        self.stitching_co_id = stitching_co_id
        self.ccf_registration_co_id = ccf_registration_co_id
        self.cell_segmentation_co_id = cell_segmentation_co_id
        self.cell_quantification_co_id = cell_quantification_co_id

    def _get_path_from_result_file(
        self,
        co_client: CodeOceanClient,
        computation_id: str,
        path_to_file: str,
    ) -> str:
        """
        Gets the path of a smartSPIM pipeline result

        Parameters
        ----------------

        co_client: CodeOceanClient
            Code Ocean Client to call the API

        computation_id: str
            Computation id of a computation in a
            capsule

        path_to_file: str
            Path to the file that has the output path
            of a result stored in a different bucket

        Returns
        ----------------
        str
            String with the path
        """

        res = co_client.get_result_file_download_url(computation_id, path_to_file)
        res = res.json()

        s3_path = None

        if "url" in res:
            response = requests.get(res["url"])
            txt_content = response.content.decode("utf-8")
            s3_path = txt_content.split(" ")[-1]

        return s3_path

    def __register_data_asset_external_bucket(
        self,
        co_client: CodeOceanClient,
        computation_state: dict,
        output_processed_filename: str,
        tags: list,
    ) -> dict:
        """
        Registers a data asset stored in a bucket

        Parameters
        ----------------

        co_client: CodeOceanClient
            Code Ocean Client to call the API

        computation_state: dict
            Dictionary with the data related to
            a computation

        output_processed_filename: str
            Filename with the output path of the dataset

        tags: list
            Tags that will be added to the data asset

        Returns
        ----------------
        dict
            Result of the Code Ocean API call to
            create a new data asset
        """

        result_created_data_asset_res = None

        s3_path = self._get_path_from_result_file(
            co_client, computation_state["id"], output_processed_filename
        )

        if s3_path:
            s3_path_splitted = [val for val in s3_path.rstrip().split("/") if len(val)]

            asset_name = s3_path_splitted[-1]
            bucket_name = s3_path_splitted[1]
            prefix = "/".join(s3_path_splitted[2:-1] + [s3_path_splitted[-1]])

            # Registering data asset from the computation
            # I'm assuming this is trying to register an asset in a public
            # bucket.

            source = Source(
                aws=Sources.AWS(
                    bucket=bucket_name,
                    prefix=prefix,
                    keep_on_external_storage=True,
                    public=True,
                )
            )
            create_data_asset_request = CreateDataAssetRequest(
                name=asset_name, tags=tags, mount=asset_name, source=source
            )

            # Registering data asset from the computation
            result_created_data_asset_res = co_client.create_data_asset(
                request=create_data_asset_request
            )
            result_created_data_asset_res = result_created_data_asset_res.json()

            logger.info(
                f"Registered result as data asset: {result_created_data_asset_res}"
            )
        else:
            logger.error(
                f"An error ocurred while reading the content file {output_processed_filename}!"
            )

        return result_created_data_asset_res

    def trigger_stitching_capsule(
        self,
        co_client: CodeOceanClient,
        params: dict,
        viewable_to_everyone: Optional[bool] = True,
    ) -> dict:
        """
        Triggers the stitching capsule

        Parameters
        ----------------
        co_client: CodeOceanClient
            Code Ocean Client to call the API

        params: dict
            Dictionary with the pipeline
            parameters

        viewable_to_everyone: Optional[bool]
            Make registered dataset available
            to everyone. Default: True

        Returns
        ----------------
        Data asset with the stitching results

        """
        tags = ["smartspim", DataLevel.RAW.value]
        bucket = self.configs["bucket"]
        prefix = self.configs["prefix"]
        mount = prefix
        # aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        # aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')

        # Since the access keys above are commented out, I'm assuming this
        # is trying to create a data asset from a public bucket
        # Creating data asset
        source = Source(
            aws=Sources.AWS(
                bucket=bucket,
                prefix=prefix,
                keep_on_external_storage=True,
                public=True,
            )
        )
        create_data_asset_request = CreateDataAssetRequest(
            name=prefix, mount=mount, source=source, tags=tags
        )
        data_asset_reg_response = co_client.create_data_asset(
            request=create_data_asset_request
        )
        response_contents = data_asset_reg_response.json()
        logger.info(f"Created data asset in Code Ocean: {response_contents}")

        if viewable_to_everyone:
            self.make_data_viewable(co_client, response_contents)

        data_asset_id = response_contents["id"]
        data_assets = [ComputationDataAsset(id=data_asset_id, mount=mount)]

        # Creating run
        run_capsule_request = RunCapsuleRequest(
            capsule_id=self.stitching_co_id,
            data_assets=data_assets,
            parameters=[
                f"/data/{prefix}/SmartSPIM",  # Input data
                f"/{params['co_folder']}/{prefix}",  # preprocessed data
                f"/{params['co_folder']}/{prefix}",  # output data
                str(params["channel"]),  # stitch channel
                str(params["resolution"]["x"]),  # Resolution in X
                str(params["resolution"]["y"]),  # Resolution in Y
                str(params["resolution"]["z"]),  # Resolution in Z
                str(16),  # CPUs
                bucket,  # Bucket
            ],
        )
        run_capsule_response = co_client.run_capsule(request=run_capsule_request)
        run_capsule_response_json = run_capsule_response.json()
        logger.info(
            f"Created run [stitching] in Code Ocean: {run_capsule_response_json}"
        )

        # Saving result as data asset once capsule finishes running
        computation_state = self.wait_for_capsule_to_finish(
            co_client=co_client, computation_id=run_capsule_response_json["id"]
        )

        # Since the datasets are being saved somewhere else, we need to create them in CO as data assets
        output_processed_filename = "output_stitching.txt"
        result_created_data_asset_res = self.__register_data_asset_external_bucket(
            co_client,
            computation_state,
            output_processed_filename,
            ["smartspim", "processed"],
        )

        # Making stitched data viewable to everyone
        if viewable_to_everyone:
            self.make_data_viewable(co_client, result_created_data_asset_res)

        return result_created_data_asset_res

    def trigger_ccf_registration(
        self,
        co_client: CodeOceanClient,
        smartspim_stitched_data_asset: dict,
        img_channel: str,
        img_scale: str = "3",
        input_zarr_directory: str = "processed/stitching/OMEZarr",
    ) -> dict:
        """
        Triggers the ccf registration capsule

        Parameters
        ----------------
        co_client: CodeOceanClient
            Code Ocean Client to call the API

        smartspim_stitched_data_asset: dict
            Dictionary with the stitched data asset
            information

        img_channel: str
            Channel for CCF registration

        img_scale: str
            Multiscale used for the CCF registration

        input_zarr_directory: str
            Directory where the image is stored

        Returns
        ----------------
        dict
            Dictionary with the capsule execution
            details
        """

        run_ccf_capsule_response = None

        if smartspim_stitched_data_asset:
            logger.info(
                f"Executing CCF registration with data asset {smartspim_stitched_data_asset}"
            )

            data_assets_ccf = [
                ComputationDataAsset(
                    id=smartspim_stitched_data_asset["id"],
                    mount=smartspim_stitched_data_asset["name"],
                )
            ]

            run_capsule_request = RunCapsuleRequest(
                capsule_id=self.ccf_registration_co_id,
                data_assets=data_assets_ccf,
                parameters=[
                    smartspim_stitched_data_asset["name"],
                    img_channel,
                    img_scale,
                    input_zarr_directory,
                ],
            )
            run_ccf_capsule_response = co_client.run_capsule(
                request=run_capsule_request
            )

            run_ccf_capsule_response = run_ccf_capsule_response.json()

            logger.info(
                f"Created run [ccf registration] in Code Ocean: {run_ccf_capsule_response}"
            )

            if "id" in run_ccf_capsule_response:
                # Saving result as data asset once capsule finishes running

                # flake8: noqa: F841
                computation_state = self.wait_for_capsule_to_finish(
                    co_client=co_client,
                    computation_id=run_ccf_capsule_response["id"],
                )

                # Saving CCF result as data asset ?? If so, fix asset name for registration
                # output_processed_filename = 'output_ccf.txt'

                # result_created_data_asset_res = self.__register_data_asset_external_bucket(
                #     co_client,
                #     computation_state,
                #     output_processed_filename,
                #     ['smartspim', 'processed', 'ccf']
                # )

            else:
                logger.error(
                    f"Error while triggering capsule: {run_ccf_capsule_response}"
                )

        else:
            logger.error(
                f"An error occurred while trigerring CCF Registration. Check data asset: {smartspim_stitched_data_asset}"
            )

        return run_ccf_capsule_response

    def trigger_cell_segmentation(
        self,
        co_client: CodeOceanClient,
        smartspim_stitched_data_asset: dict,
        img_channel: str,
        img_scale: str = "0",
        chunk_size: str = "500",
    ) -> dict:
        """
        Triggers the cell segmentation capsule

        Parameters
        ----------------
        co_client: CodeOceanClient
            Code Ocean Client to call the API

        smartspim_stitched_data_asset: dict
            Dictionary with the stitched data asset
            information

        img_channel: str
            Channel for CCF registration

        img_scale: str
            Multiscale used for the CCF registration

        Returns
        ----------------
        dict
            Dictionary with the capsule execution
            details
        """

        run_seg_capsule_response = None

        if smartspim_stitched_data_asset:
            logger.info(
                f"Executing cell segmentation with data asset {smartspim_stitched_data_asset}"
            )

            data_assets_seg = [
                ComputationDataAsset(
                    id=smartspim_stitched_data_asset["id"],
                    mount=smartspim_stitched_data_asset["name"],
                )
            ]

            run_capsule_request = RunCapsuleRequest(
                capsule_id=self.cell_segmentation_co_id,
                data_assets=data_assets_seg,
                parameters=[
                    smartspim_stitched_data_asset["name"],
                    img_channel,
                    img_scale,
                    chunk_size,
                    0,  # start signal
                    -1,  # end signal
                    self.configs["bucket"],
                ],
            )
            run_seg_capsule_response = co_client.run_capsule(
                request=run_capsule_request
            )

            run_seg_capsule_response = run_seg_capsule_response.json()

            logger.info(
                f"Created run [ccf registration] in Code Ocean: {run_seg_capsule_response}"
            )

            if "id" in run_seg_capsule_response:
                # Saving result as data asset once capsule finishes running
                computation_state = self.wait_for_capsule_to_finish(
                    co_client=co_client,
                    computation_id=run_seg_capsule_response["id"],
                )

            else:
                logger.error(
                    f"Error while triggering capsule: {run_seg_capsule_response}"
                )

        else:
            logger.error(
                f"An error occurred while trigerring cell segmentation. Check data asset: {smartspim_stitched_data_asset}"
            )

        return run_seg_capsule_response

    def trigger_cell_quantification(
        self,
        co_client: CodeOceanClient,
        smartspim_stitched_data_asset: dict,
        channel_name: str,
        save_path: str,
        intermediate_folder: str = "processed/stitching/OMEZarr",
        downsample_res: str = "3",
        reference_microns_ccf: str = "25",
    ):
        """
        Triggers the quantification capsule

        Parameters
        ----------------
        co_client: CodeOceanClient
            Code Ocean Client to call the API

        smartspim_stitched_data_asset: dict
            Dictionary with the stitched data asset
            information

        channel_name: str
            Channel for cell quantification. CCF registration
            and cell segmentation steps must have been executed
            before

        save_path: str
            Path where the capsule with write its outputs

        intermediate_folder: str
            Intermediate folder where the results are
            for the smartspim datasets. It's important since
            we have multiple versions of the pipeline.
            Default: "processed/stitching/OMEZarr"

        downsample_res: str
            Resolution that was used in the
            CCF registration

        reference_microns_ccf: str
            Reference size in microns for the CCF
            Atlas
        Returns
        ----------------
        dict
            Dictionary with the capsule execution
            details
        """

        run_cell_quant_capsule_response = None

        if smartspim_stitched_data_asset:
            logger.info(
                f"Executing quantification with data asset {smartspim_stitched_data_asset}"
            )

            data_assets_ccf = [
                {
                    "id": smartspim_stitched_data_asset["id"],
                    "mount": smartspim_stitched_data_asset["name"],
                }
            ]
            data_assets_ccf = [
                ComputationDataAsset(
                    id=smartspim_stitched_data_asset["id"],
                    mount=smartspim_stitched_data_asset["name"],
                )
            ]
            run_capsule_request = RunCapsuleRequest(
                capsule_id=self.cell_quantification_co_id,
                data_assets=data_assets_ccf,
                parameters=[
                    smartspim_stitched_data_asset,
                    channel_name,
                    save_path,
                    intermediate_folder,
                    downsample_res,
                    reference_microns_ccf,
                ],
            )
            run_cell_quant_capsule_response = co_client.run_capsule(
                request=run_capsule_request
            )

            run_cell_quant_capsule_response = run_cell_quant_capsule_response.json()

            logger.info(
                f"Created run [cell quantification] in Code Ocean: {run_cell_quant_capsule_response}"
            )

            if "id" in run_cell_quant_capsule_response:
                # Saving result as data asset once capsule finishes running

                # flake8: noqa: F841
                computation_state = self.wait_for_capsule_to_finish(
                    co_client=co_client,
                    computation_id=run_cell_quant_capsule_response["id"],
                )

            else:
                logger.error(
                    f"Error while triggering capsule: {run_cell_quant_capsule_response}"
                )

        else:
            logger.error(
                f"An error occurred while trigerring CCF Registration. Check data asset: {smartspim_stitched_data_asset}"
            )

        return run_cell_quant_capsule_response

    def run_job(self):
        """
        Executes the SmartSPIM pipeline
        """
        codeocean_domain = os.getenv("CODEOCEAN_DOMAIN")
        co_client = CodeOceanClient(
            domain=codeocean_domain, token=os.getenv("CUSTOM_KEY")
        )

        if "stitching" not in self.configs:
            logger.info("Pipeline can't be triggered without stitching")
            exit(1)

        # Trigger stitching
        result_created_data_asset_res = self.trigger_stitching_capsule(
            co_client, self.configs["stitching"]
        )

        # flake8: noqa: F841
        if (
            "registration" in self.configs
            and "channels" in self.configs["registration"]
        ):
            for channel_name in self.configs["registration"]["channels"]:
                logger.info(f"Registering channel {channel_name}")

                result_data_asset_registration = self.trigger_ccf_registration(
                    co_client,
                    result_created_data_asset_res,
                    channel_name,
                    self.configs["registration"]["input_scale"],
                    "processed/stitching/OMEZarr",
                )

        # Trigger Cell Segmentation
        if (
            "segmentation" in self.configs
            and "channels" in self.configs["segmentation"]
        ):
            for channel_name in self.configs["registration"]["channels"]:
                logger.info(f"Cell segmentation in channel {channel_name}")

                result_cell_segmentation = self.trigger_cell_segmentation(
                    co_client,
                    result_created_data_asset_res,
                    channel_name,
                    self.configs["segmentation"]["input_scale"],
                    self.configs["segmentation"]["chunksize"],
                )

        if "registration" in self.configs and "segmentation" in self.configs:
            # Executing only when registration and segmentation are executed
            channels_ccf = set(self.configs["registration"]["channels"])
            channels_cell = set(self.configs["registration"]["channels"])

            # Getting channels that are common for the two configurations
            common_channels = channels_ccf & channels_cell

            for channel_name in common_channels:
                logger.info(f"Running quantification in channel {channel_name}")

                result_cell_quantification = self.trigger_cell_quantification(
                    co_client=co_client,  # Code ocean client
                    smartspim_stitched_data_asset=result_created_data_asset_res,  # Stitched dataset
                    channel_name=channel_name,  # Channel name
                    save_path="/results/",  # output folder
                    intermediate_folder="processed/stitching/OMEZarr",
                    downsample_res="3",
                    reference_microns_ccf="25",
                )
