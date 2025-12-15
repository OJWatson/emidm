def test_imports_for_examples_and_slides():
    from emidm.sir import run_sir, run_model_with_replicates, plot_model_outputs

    assert callable(run_sir)
    assert callable(run_model_with_replicates)
    assert callable(plot_model_outputs)
