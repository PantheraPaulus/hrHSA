def test_core_imports():
    import hsa
    from hsa import FeatureSpec
    from hsa.rsf import fit_rsf
    from hsa.movement import prepare_trajectory_data

    assert hsa is not None
    assert FeatureSpec is not None
    assert fit_rsf is not None
    assert prepare_trajectory_data is not None
