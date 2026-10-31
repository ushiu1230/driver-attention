from datetime import datetime
from email.message import EmailMessage
import logging
import os
from pathlib import Path
import smtplib

import torch
from torch.utils.data import DataLoader

from dataset.dataset import DG360LiteDataset
from models import SingleFrameSaliencyNet
from utils.metrics import SaliencyLoss
from utils.tflogger import TFLogger


def notify_training_error(error, log_path):
    email_user = os.environ.get("DRIVER_ATTENTION_EMAIL_USER")
    email_password = os.environ.get("DRIVER_ATTENTION_EMAIL_PASSWORD")
    if not email_user or not email_password:
        logging.warning(
            "Email alert not sent. Set DRIVER_ATTENTION_EMAIL_USER and "
            "DRIVER_ATTENTION_EMAIL_PASSWORD."
        )
        return

    email_host = os.environ.get("DRIVER_ATTENTION_EMAIL_HOST", "smtp.gmail.com")
    try:
        email_port = int(os.environ.get("DRIVER_ATTENTION_EMAIL_PORT", "465"))
    except ValueError:
        logging.error("DRIVER_ATTENTION_EMAIL_PORT must be an integer.")
        return
    recipient = "ushiu1230@gmail.com"
    message = EmailMessage()
    message["Subject"] = "Driver-attention training failed"
    message["From"] = email_user
    message["To"] = recipient
    message.set_content(
        "Training stopped because of an error.\n"
        f"Error: {type(error).__name__}: {error}\n"
        f"The full training log is attached: {log_path}"
    )
    try:
        message.add_attachment(
            log_path.read_bytes(),
            maintype="text",
            subtype="plain",
            filename=log_path.name,
        )
    except OSError:
        logging.exception("Could not attach the training log.")

    try:
        with smtplib.SMTP_SSL(email_host, email_port, timeout=15) as server:
            server.login(email_user, email_password)
            server.send_message(message)
        logging.info("Training error email sent to %s.", recipient)
    except (OSError, smtplib.SMTPException) as notification_error:
        logging.exception(
            "Could not send training error email: %s", notification_error
        )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    logger=None,
    phase="train",
    step_offset=0,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)  # [B, T, 3, H, W]
            target = batch["sal"].to(device)  # [B, 1, H, W]

            # Bỏ mẫu có target rỗng để tránh chuẩn hóa phân phối không hợp lệ.
            valid = target.flatten(start_dim=1).sum(dim=1) > 1e-8
            if not valid.any():
                continue

            rgb = rgb[valid]
            target = target[valid]

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(rgb)
            loss = criterion(prediction, target)

            if training:
                loss.backward()
                optimizer.step()

            if logger is not None:
                logger.log(
                    {"kld": loss.detach()},
                    f"batch/{phase}",
                    step_offset + batch_index,
                )

            count = target.size(0)
            total_loss += loss.detach().item() * count
            total_samples += count

    if total_samples == 0:
        raise RuntimeError("Epoch không có mẫu saliency hợp lệ.")

    return total_loss / total_samples


def main():
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("results") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("Training log: %s", log_path.resolve())

    logger = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_ds = DG360LiteDataset(
            root="dataset/drivergaze360_subset/dataset/train",
            T=16,
            frame_stride=1,
            clip_step=8,
            img_size=(224, 1120),
        )
        val_ds = DG360LiteDataset(
            root="dataset/drivergaze360_subset/dataset/val",
            T=16,
            frame_stride=1,
            clip_step=8,
            img_size=(224, 1120),
        )

        train_loader = DataLoader(
            train_ds, batch_size=2, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=2, shuffle=False, num_workers=2
        )

        model = SingleFrameSaliencyNet(pretrained=True).to(device)
        criterion = SaliencyLoss(lambda_cc=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        best_val_loss = float("inf")
        logger = TFLogger(run_id=run_id, save_dir="tensorboard")
        logging.info("TensorBoard log directory: %s", logger.save_dir)

        for epoch in range(20):
            train_loss = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                logger=logger,
                phase="train",
                step_offset=epoch * len(train_loader),
            )
            val_loss = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                logger=logger,
                phase="val",
                step_offset=epoch * len(val_loader),
            )

            logging.info(
                f"Epoch {epoch + 1:02d}: "
                f"train KLD={train_loss:.4f}, val KLD={val_loss:.4f}"
            )
            logger.log_epoch(train_loss, val_loss, {}, epoch + 1)
            logger.log(
                {"learning_rate": optimizer.param_groups[0]["lr"]},
                "epoch",
                epoch + 1,
            )
            logger.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    "best_single_frame.pt",
                )
    except Exception as error:
        logging.exception("Training stopped because of an error.")
        notify_training_error(error, log_path.resolve())
        raise
    finally:
        if logger is not None:
                logger.close()


if __name__ == "__main__":
    main()