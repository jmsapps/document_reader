import json
from typing import Any

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.credentials import TokenCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError, ResourceExistsError

from src.auth.iam import IAM


class AzureStorageAccountService:
    """Basic Azure Blob Storage account service."""

    def __init__(self, endpoint: str, api_key: str | None) -> None:
        credential = self._build_credential(api_key)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.client = BlobServiceClient(account_url=self.endpoint, credential=credential)

    @staticmethod
    def _build_credential(api_key: str | None) -> str | TokenCredential:
        if api_key:
            return api_key

        return IAM().get_credential()

    def test_connection(self) -> dict[str, Any]:
        return self.client.get_service_properties()  # type: ignore[no-any-return]

    def ensure_container(self, container_name: str) -> None:
        container_client = self.client.get_container_client(container_name)

        try:
            container_client.create_container()
        except ResourceExistsError:
            pass

    def delete_container_if_exists(self, container_name: str) -> None:
        container_client = self.client.get_container_client(container_name)
        try:
            container_client.delete_container()
        except ResourceNotFoundError:
            return
        except HttpResponseError as exc:
            message = (str(exc) or "").lower()
            if "containernotfound" in message or "container does not exist" in message:
                return
            raise

    def upload_bytes(
        self,
        *,
        container_name: str,
        blob_name: str,
        data: bytes,
        content_type: str | None = None,
        overwrite: bool = True,
    ) -> str:
        self.ensure_container(container_name)
        blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)
        kwargs: dict[str, Any] = {"overwrite": overwrite}
        if content_type:
            kwargs["content_settings"] = ContentSettings(content_type=content_type)
        blob_client.upload_blob(data, **kwargs)
        return blob_client.url

    def upload_text(
        self,
        *,
        container_name: str,
        blob_name: str,
        text: str,
        overwrite: bool = True,
    ) -> str:
        return self.upload_bytes(
            container_name=container_name,
            blob_name=blob_name,
            data=text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            overwrite=overwrite,
        )

    def upload_json(
        self,
        *,
        container_name: str,
        blob_name: str,
        payload: Any,
        overwrite: bool = True,
    ) -> str:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self.upload_bytes(
            container_name=container_name,
            blob_name=blob_name,
            data=data,
            content_type="application/json; charset=utf-8",
            overwrite=overwrite,
        )

    def download_bytes(self, *, container_name: str, blob_name: str) -> bytes:
        blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)
        stream = blob_client.download_blob()
        return stream.readall()

    def blob_exists(self, *, container_name: str, blob_name: str) -> bool:
        blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)

        try:
            return blob_client.exists()
        except (ResourceNotFoundError, HttpResponseError):
            return False

    def list_blobs(self, *, container_name: str, prefix: str | None = None) -> list[str]:
        container_client = self.client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]

    def get_blob_url(self, *, container_name: str, blob_name: str) -> str:
        blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)
        return blob_client.url
