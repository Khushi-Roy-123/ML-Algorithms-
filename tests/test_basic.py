def test_import_and_version():
    import jklearn

    assert hasattr(jklearn, "__version__")
    assert isinstance(jklearn.__version__, str)
