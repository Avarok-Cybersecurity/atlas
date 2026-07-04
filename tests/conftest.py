def pytest_addoption(parser):
    parser.addoption("--update-baselines", action="store_true", default=False,
                      help="write current measurements as the new baseline instead of "
                           "asserting against it — only right before merge")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "multirank: uses BOTH physical GPU nodes at once (EP=2/TP=2/TP+EP); "
        "cannot run concurrently with single-GPU head/worker groups",
    )
