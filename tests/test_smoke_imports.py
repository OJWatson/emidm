def test_imports_for_examples_and_slides():
    from emidm.sir import run_sir_simulation
    from emidm.safir import run_safir_simulation
    from emidm import to_dataframe, DiffConfig, make_diff_safir_model

    assert callable(run_sir_simulation)
    assert callable(run_safir_simulation)
    assert callable(to_dataframe)
    assert DiffConfig is not None
    assert callable(make_diff_safir_model)
