import torch
import torchio as tio


def _get_input_and_target(batch, device):
    """
    Extract and concatenate multi-modal inputs (T1N, T2W, T2F) and target (T1C).
    """
    t1n = batch['t1n'][tio.DATA]
    t2w = batch['t2w'][tio.DATA]
    t2f = batch['t2f'][tio.DATA]
    x = torch.cat([t1n, t2w, t2f], dim=1).to(device)
    target = batch['t1c'][tio.DATA].to(device)
    return x, target


def _compute_flow_loss(x, t1c, model, sigma_s, device):
    """
    Compute standard rectified flow loss
    """
    eps = torch.randn_like(x)
    z_0 = x + sigma_s * eps
    t = torch.rand(t1c.size(0), device=device)
    tb = t.view(-1, 1, 1, 1, 1)
    z_t = tb * t1c + (1 - tb) * z_0
    target = t1c - z_0
    v_pred = model(z_t, t)
    return torch.nn.MSELoss()(v_pred, target)


def _compute_cond_flow_loss(x, t1c, model, device):
    """
    Compute conditional rectified flow loss
    """
    z_0 = torch.randn_like(t1c)
    t = torch.rand(t1c.size(0), device=device)
    tb = t.view(-1, 1, 1, 1, 1)
    z_t = tb * t1c + (1 - tb) * z_0
    inp = torch.cat([z_t, x], dim=1)
    target = t1c - z_0
    v_pred = model(inp, t)
    return torch.nn.MSELoss()(v_pred, target)


def compute_loss_pm(model, batch, device):
    """
    Posterior-mean MSE loss using all three input modalities.
    """
    x, t1c = _get_input_and_target(batch, device)
    return torch.nn.MSELoss()(model(x), t1c)


def compute_loss_rf(model, batch, frozen_pm, sigma_s, device):
    """
    Rectified flow loss fine-tuning a frozen posterior-mean model:
    - Use frozen_pm(x) as base state, then apply standard flow loss.
    """
    x, t1c = _get_input_and_target(batch, device)
    with torch.no_grad():
        pred_pm = frozen_pm(x)

    return _compute_flow_loss(pred_pm, t1c, model, sigma_s, device)


def compute_loss_rf_flow_from_x_t1n(model, batch, sigma_s, device):
    """
    Rectified flow baseline loss directly from T1N input (no PM):
    - Treat t1n as base state, then standard flow loss.
    """
    t1n = batch['t1n'][tio.DATA].to(device)
    t1c = batch['t1c'][tio.DATA].to(device)
    return _compute_flow_loss(t1n, t1c, model, sigma_s, device)


def compute_loss_rf_cond_x(model, batch, device):
    """
    Conditional rectified flow loss conditioned on all three modalities:
    - Stack inputs as conditioning x, then apply conditional flow loss.
    """
    x, t1c = _get_input_and_target(batch, device)
    return _compute_cond_flow_loss(x, t1c, model, device)


def compute_loss_rf_cond_yhat_pm(model, batch, frozen_pm, device):
    """
    Conditional rectified flow loss conditioned on PM prediction:
    - Use frozen_pm(x) as conditioning, then standard conditional flow loss.
    """
    x, t1c = _get_input_and_target(batch, device)

    with torch.no_grad():
        pred_pm = frozen_pm(x)

    return _compute_cond_flow_loss(pred_pm, t1c, model, device)


def compute_loss_pm_t1n(model, batch, device):
    """
    Posterior-mean MSE loss using only T1N input.
    """
    x = batch['t1n'][tio.DATA].to(device)
    target = batch['t1c'][tio.DATA].to(device)
    return torch.nn.MSELoss()(model(x), target)


def compute_loss_rf_t1n(model, batch, frozen_pm, sigma_s, device):
    """
    Rectified flow loss fine-tuning a T1N-only posterior-mean model.
    """
    x = batch['t1n'][tio.DATA].to(device)
    t1c = batch['t1c'][tio.DATA].to(device)

    with torch.no_grad():
        pred_pm = frozen_pm(x)

    return _compute_flow_loss(pred_pm, t1c, model, sigma_s, device)
