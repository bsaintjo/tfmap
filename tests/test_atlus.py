from tfmap import Atlus

def test_load():
    atlus = Atlus.from_map_filepath("data/test.map")

    assert len(atlus.spectra()) == 6732