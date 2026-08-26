from perception_rt.system_info import collect_system_info


def test_collect_system_info_returns_required_fields() -> None:
    info = collect_system_info()

    assert "pytorch" in info
    assert "torchvision" in info
    assert "cuda_available" in info
    assert isinstance(info["cuda_available"], bool)
