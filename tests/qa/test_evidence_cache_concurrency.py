from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from syntide.generate.qa import evidence


def test_concurrent_cache_writers_publish_independent_staging_files(
    tmp_path, monkeypatch
):
    cache = evidence.EvidenceCache(tmp_path)
    rendezvous = Barrier(4)
    replace = evidence.os.replace

    def synchronized_replace(source, destination):
        rendezvous.wait(timeout=10)
        replace(source, destination)

    monkeypatch.setattr(evidence.os, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: cache.put("same-key", []), range(4)))
    assert cache.get("same-key") == []
    assert not list(tmp_path.rglob("*.partial"))
