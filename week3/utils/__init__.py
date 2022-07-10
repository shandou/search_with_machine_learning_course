import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s-%(filename)s->%(funcName)s:-%(levelname)s] "
        "%(message)s"
    ),
)
logger: logging.Logger = logging.getLogger(__name__)
