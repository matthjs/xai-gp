from loguru import logger
import time

def log_training_start(model_type, num_epochs, num_samples):
    """Log the start of a training process."""
    logger.info(f"Training {model_type} model for {num_epochs} epochs on {num_samples} samples...")
    return time.time()


def log_epoch_stats(epoch, num_epochs, epoch_loss, data_loader_length, epoch_start_time):
    """Log statistics for a single training epoch."""
    epoch_time = time.time() - epoch_start_time
    avg_epoch_loss = epoch_loss / data_loader_length
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs} - "
        f"Loss: {avg_epoch_loss:.4f} - "
        f"Time: {epoch_time:.2f}s"
    )
    return avg_epoch_loss


def log_training_end(start_time, epoch_loss, data_loader_length):
    """Log the end of a training process."""
    total_time = time.time() - start_time
    logger.info(
        f"Training completed in {total_time:.2f}s. "
        f"Final Loss: {epoch_loss / data_loader_length:.4f}"
    )