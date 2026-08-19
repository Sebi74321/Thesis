"""Reporting for the upstream DP-CGANS privacy accountant."""

from __future__ import annotations

from typing import Any


def upstream_privacy_accounting(
    model, train_rows: int, model_config: dict
) -> dict[str, Any]:
    """Capture upstream estimates without presenting them as a formal DP guarantee."""
    batch_size = int(model_config["batch_size"])
    epochs = int(model_config["epochs"])
    discriminator_steps = int(model_config.get("discriminator_steps", 1))
    steps_per_epoch = max(train_rows // batch_size, 1)
    upstream_steps = max(epochs - 1, 0) * steps_per_epoch
    actual_discriminator_updates = epochs * steps_per_epoch * discriminator_steps
    result: dict[str, Any] = {
        "privacy_claim_status": "upstream_privacy_estimate_unverified",
        "private": True,
        "noise_multiplier": 1.0,
        "delta": 2e-6,
        "sampling_probability": batch_size / train_rows,
        "train_rows": train_rows,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "discriminator_steps": discriminator_steps,
        "upstream_accountant_steps": upstream_steps,
        "actual_discriminator_updates": actual_discriminator_updates,
        "upstream_reported_epsilon": getattr(model, "upstream_reported_epsilon", None),
        "limitation": (
            "The upstream mechanism uses parameter-gradient hooks and weight clipping, not "
            "conventional per-example DP-SGD clipping. Values are implementation-specific estimates."
        ),
    }
    try:
        from dp_cgans.functions.rdp_accountant import compute_rdp, get_privacy_spent

        orders = [1 + x / 10.0 for x in range(1, 100)]
        for label, steps in (
            ("epsilon_recomputed_upstream_steps", upstream_steps),
            ("epsilon_recomputed_actual_updates", actual_discriminator_updates),
        ):
            rdp = compute_rdp(
                q=result["sampling_probability"],
                noise_multiplier=result["noise_multiplier"],
                steps=steps,
                orders=orders,
            )
            epsilon, _, order = get_privacy_spent(
                orders, rdp, target_delta=result["delta"]
            )
            result[label] = float(epsilon)
            result[f"{label}_optimal_order"] = float(order)
    except Exception as exc:
        result["accountant_error"] = f"{type(exc).__name__}: {exc}"
    return result
