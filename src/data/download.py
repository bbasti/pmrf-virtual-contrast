import os
import shutil
from synapseclient import Synapse


def download_and_unzip(synapse_ids, destination):
    """
    Download & unzip entity archives from Synapse.
    """
    auth_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if auth_token is None:
        raise RuntimeError("Set SYNAPSE_AUTH_TOKEN env var.")
    syn = Synapse()
    syn.login(authToken=auth_token)
    os.makedirs(destination, exist_ok=True)

    for sid in synapse_ids:
        print(f"[download] {sid}")
        entity = syn.get(sid, downloadLocation=destination)
        archive = entity.path
        shutil.unpack_archive(archive, destination)
        os.remove(archive)
        # clear cache
        shutil.rmtree(os.path.expanduser("~/.synapseCache"), ignore_errors=True)
