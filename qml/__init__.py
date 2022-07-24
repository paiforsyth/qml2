from hypothesis import settings

qml_hypothesis_profile = "qml_hypothesis_profile"
settings.register_profile(qml_hypothesis_profile, deadline=None)
settings.load_profile(qml_hypothesis_profile)  # avoid failing tests with inconsistent timing
