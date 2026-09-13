import pytest

from hsa.compute import ExecutionConfig


def test_execution_config_accepts_partition_graph_policy():
    config = ExecutionConfig(point_graph_partitions=100)
    assert config.point_graph_partitions == 100

    with pytest.raises(ValueError, match="point_graph_partitions"):
        ExecutionConfig(point_graph_partitions=0)
