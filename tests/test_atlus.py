from tfmap import Atlus


def test_load():
    atlus = Atlus.from_map_filepath("data/test.map")

    assert len(atlus.image_extent()) == 4
    assert atlus.image_extent()[0] < atlus.image_extent()[1]
    assert atlus.image_extent()[2] < atlus.image_extent()[3]
    assert len(atlus.spectra()) == 6732
