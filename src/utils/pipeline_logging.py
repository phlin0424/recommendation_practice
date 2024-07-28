import logging


def configure_logging():
    """Configure the logging format
    Available format:
    %(asctime)s: Human-readable time when the log record was created.
    %(created)f: Time when the log record was created (as a float).
    %(filename)s: Filename portion of pathname.
    %(funcName)s: Name of the function containing the logging call.
    %(levelname)s: Text logging level for the message ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    %(lineno)d: Source line number where the logging call was issued.
    %(message)s: The logged message.
    %(module)s: Module (name portion of filename).
    %(name)s: Name of the logger used to log the call.
    %(pathname)s: Full pathname of the source file where the logging call was issued.
    %(process)d: Process ID.
    %(processName)s: Process name.
    %(thread)d: Thread ID.
    %(threadName)s: Thread name.
    Returns:
        _type_: _description_
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)s.%(name)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    # Test the logging format configuration:
    logger = configure_logging()

    # Example usage
    def example_function():
        logger.info("This is an info message from example_function.")

    class ExampleClass:
        def example_method(self):
            logger.info("This is an info message from example_method in ExampleClass.")

    example_function()

    example_instance = ExampleClass()
    example_instance.example_method()
