from tests.support.legacy_loader import FakeHarness, load_legacy_main


def test_import_uses_test_doubles_without_credentials_or_downloads() -> None:
    harness = FakeHarness()

    module = load_legacy_main(harness)

    assert module.client is harness.client
    assert harness.api_keys == [None]
    assert harness.downloads == ["punkt"]
    assert harness.dotenv_load_count == 1
