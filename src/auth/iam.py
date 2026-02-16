from azure.identity import DefaultAzureCredential


class IAM:
    """Small IAM wrapper to keep credential logic in one place."""

    def get_credential(self):
        return DefaultAzureCredential(exclude_interactive_browser_credential=False)
